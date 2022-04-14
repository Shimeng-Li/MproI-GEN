import torch
import os
import rdkit
import sys
import argparse
from rdkit import Chem
from model_storage import ModelStorage
from utils import add_train_args, set_seed, read_smiles_csv, get_dataset

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODEL = ModelStorage()

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models trainer script', description='available models')
    add_train_args(MODEL.get_model_train_parser(MODEL.get_model_names())(subparsers.add_parser(MODEL.get_model_names())))
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    if config.train_load is None:
        train_data = get_dataset('train')
    else:
        train_data = read_smiles_csv(config.train_load)
    if config.val_load is None:
        val_data = get_dataset('test')
    else:
        val_data = read_smiles_csv(config.val_load)
    trainer = MODEL.get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), 'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = MODEL.get_model_class(model)(vocab, config).to(device)
    trainer.fit(model, train_data, val_data)
    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)

    # model_ft = model.to(device)
    # trainer.fine_tuning(model_ft, ft_train_data, ft_val_data)
    # model_ft = model_ft.to('cpu')
    # torch.save(model_ft.state_dict(), config.model_save)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
