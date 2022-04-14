import re
import os
import torch
import random
import argparse
import numpy as np
import scipy.sparse
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from model_metrics.SA_Score import sascorer
from model_metrics.NP_Score import npscorer
from functools import partial
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from collections import Counter
from rdkit.Chem import MACCSkeys
from multiprocessing import Pool
from rdkit.Chem import Descriptors
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from collections import UserList, defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan


# model_utils
class MetaTrainer(ABC):
    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return model.device

    def get_dataloader(self, model, data, collate_fn=None, shuffle=True):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        return DataLoader(data, batch_size=self.config.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)

    def get_ft_dataloader(self, model, data, collate_fn=None, shuffle=True):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        return DataLoader(data,
                          batch_size=self.config.ft_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)


    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def get_vocabulary(self, data):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]
        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string


class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, slice):
            return Logger(self.data[key])
        ldata = self.sdata[key]
        if isinstance(ldata[0], dict):
            return Logger(ldata)
        return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


# utils
def set_torch_seed_to_all_gens(_):
    seed = torch.initial_seed() % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)


def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load', type=str, help='Input data in csv format to train')
    common_arg.add_argument('--seed', type=int, default=0, help='Random state')
    common_arg.add_argument('--val_load', type=str, help="Input data in csv format to validation")
    # common_arg.add_argument('--model_save', type=str, required=True, help='Where to save the model')
    common_arg.add_argument('--save_frequency', type=int, default=20, help='How often to save the model')
    common_arg.add_argument('--log_file', type=str, required=False, help='Where to save the log')
    # common_arg.add_argument('--config_save', type=str, required=True, help='Where to save the config')
    common_arg.add_argument('--vocab_save', type=str, help='Where to save the vocab')
    common_arg.add_argument('--vocab_load', type=str, help='Where to load the vocab; ' 'otherwise it will be evaluated')
    common_arg.add_argument('--loss_save', type=str, help='Where to save the loss')
    common_arg.add_argument('--running_loss_save', type=str, help='Where to save the running loss')
    return parser


def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError('Wrong device format: {}'.format(arg))
        if arg != 'cpu':
            splited_device = arg.split(':')
            if (not torch.cuda.is_available()) or (
                    len(splited_device) > 1 and int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError('Wrong device: {} is not available'.format(arg))
        return arg


def add_fine_tuning_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--seed', type=int, default=0, help='Random state')
    common_arg.add_argument('--ft_lr', type=float, default=1e-4, help='Learning rate')
    common_arg.add_argument('--ft_model_save', type=str, required=True, help='Where to save the model')
    common_arg.add_argument('--ft_save_frequency', type=int, default=100, help='How often to save the model')
    common_arg.add_argument('--ft_log_file', type=str, required=False, help='Where to save the log')
    common_arg.add_argument('--ft_config_save', type=str, required=True, help='Where to save the config')
    common_arg.add_argument('--ft_vocab_save', type=str, help='Where to save the vocab')
    common_arg.add_argument('--ft_loss_save', type=str, help='Where to save the loss')
    common_arg.add_argument('--ft_running_loss_save', type=str, help='Where to save the running loss')
    common_arg.add_argument('--ft_step_size', type=int, default=5, help='Period of learning rate decay')
    common_arg.add_argument('--ft_gamma', type=int, default=0.1, help='Period of learning rate decay')
    common_arg.add_argument('--ft_epochs', type=int, default=50, help='Epochs of Fine Tuning')
    common_arg.add_argument('--pretrained_model_load', type=str, help='Where to load the model')
    common_arg.add_argument('--pretrained_config_load', type=str, help='Where to load the config')
    common_arg.add_argument('--pretrained_vocab_load', type=str, help='Where to load the vocab')
    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load', type=str, required=True, help='Where to load the model')
    common_arg.add_argument('--config_load', type=str, required=True, help='Where to load the config')
    common_arg.add_argument('--vocab_load', type=str, required=True, help='Where to load the vocab')
    common_arg.add_argument('--seed', type=int, default=0, help='Random state')
    common_arg.add_argument('--n_samples', type=int, required=True, help='Number of samples to sample')
    common_arg.add_argument('--gen_save', type=str, required=True, help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch", type=int, default=32, help="Size of batch")
    common_arg.add_argument("--max_len", type=int, default=100, help="Max of length of SMILES")
    return parser


def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(split='train'):
    """
    Loads MOSES dataset

    Arguments:
        split (str or list): split to load. If str, must be
            one of: 'train', 'test', 'test_scaffolds'. If
            list, will load all splits from the list.
            None by default---loads all splits

    Returns:
        dict with splits. Keys---split names, values---lists
        of SMILES strings.
    """
    AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f"Unknown split {split}. "f"Available splits: {AVAILABLE_SPLITS}")
    base_path = os.path.dirname(__file__)
    print('base_path is:', base_path)
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f"Unknown split {split}. "f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, 'data', split + '.csv.gz')
    print('path is: ', path)
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')


# metrics_utils
_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, 'model_metrics/mcf.csv'))
_pains = pd.read_csv(os.path.join(_base_dir, 'model_metrics/wehi_pains.csv'), names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in _mcf.append(_pains, sort=True)['smarts'].values]


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def mol_passes_filters(mol, allowed=None, isomericSmiles=False):
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(len(x) >= 6 for x in ring_info.AtomRings()):
        return False
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    # h_mol = Chem.AddHs(mol)
    # if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
    #     return False
    mol_weight = Descriptors.MolWt(mol)
    if mol_weight < 200 or mol_weight > 800:
        return False
    logP = Chem.Crippen.MolLogP(mol)  # -5 < XlogP <= 6.5
    if logP > 6.5 or logP <-5:
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True


def fingerprint(smiles_or_mol, fp_type='morgan', dtype=None, morgan__r=2, morgan__n=2048, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)
    fps = mapper(n_jobs)(partial(fingerprint, *args, **kwargs), smiles_mols_array)
    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scaffold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def get_statistics(split='test'):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'data', split + '_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()


def process_molecule(mol_row, isomeric):
    # mol_row = mol_row.decode('utf-8')
    smiles, _id = mol_row.split()
    if not mol_passes_filters(smiles):
        return None
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomeric)
    return _id, smiles
