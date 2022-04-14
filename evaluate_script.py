from model_storage import ModelStorage
import argparse
from metrics import get_all_metrics, get_all_descriptions
from utils import read_smiles_csv
import pandas as pd
from utils import get_mol
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import numpy as np

def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='../../Dataset/PretrainedData/pretrained_zinc_500w/pretrained_zinc_500w_v3/test.csv', required=False, help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path', type=str, default='../../Dataset/PretrainedData/pretrained_zinc_500w/pretrained_zinc_500w_v3/test_scaffold.csv', required=False, help='Path to scaffold test molecules csv')
    parser.add_argument('--train_path', type=str, default='../../Dataset/FineTuningData/sars-cov-2/pos_data.csv', required=False, help='Path to train molecules csv')
    parser.add_argument('--gen_path', type=str, default='../../Result/generation_performance/ft_model_generation_performance/generated_chemicals.csv', help='Directory for sampled molecules')
    parser.add_argument('--metrics_path', type=str, default='../../Result/generation_performance/ft_model_generation_performance/generated_performance_v2.csv', help='Path to output file with metrics')
    parser.add_argument('--ks', '--unique_k', nargs='+', default=[1000, 10000], type=int, help='Number of molecules to calculate uniqueness at.' 'Multiple values are possible. Defaults to ' '--unique_k 1000 10000')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of processes to run metrics')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device id (`cpu` or `cuda:n`)')
    return parser

MODEL = ModelStorage()
parser = get_eval_parser()
common_config = parser.parse_known_args()[0]
print('Computing metrics...')
eval_parser = get_eval_parser()
eval_args = ['--gen_path', common_config.gen_path,
             '--n_jobs', str(common_config.n_jobs),
             '--device', common_config.device]

train_path = common_config.train_path
test_path = common_config.test_path
test_scaffolds_path = common_config.test_scaffolds_path
if test_path is not None:
    eval_args.extend(['--test_path', test_path])

if test_scaffolds_path is not None:
    eval_args.extend(['--test_scaffolds_path', test_scaffolds_path])

if train_path is not None:
    eval_args.extend(['--train_path', train_path])

model_name = MODEL.get_model_names()
eval_config = eval_parser.parse_args(eval_args)

if eval_config.test_path:
    test = read_smiles_csv(eval_config.test_path)

if eval_config.train_path is not None:
    train = read_smiles_csv(eval_config.train_path)

if eval_config.test_scaffolds_path is not None:
    test_scaffolds = read_smiles_csv(eval_config.test_scaffolds_path)

gen = read_smiles_csv(eval_config.gen_path)
metrics = get_all_metrics(gen=gen, k=eval_config.ks, n_jobs=eval_config.n_jobs, device=eval_config.device, test_scaffolds=test_scaffolds, test=test, train=train)

table = pd.DataFrame([metrics]).T
table.to_csv(common_config.metrics_path, header=False)
