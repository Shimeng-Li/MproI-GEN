import os
import argparse
import sys
import pandas as pd
from model_storage import ModelStorage
import torch
from utils import add_sample_args, set_seed
from tqdm.auto import tqdm

def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000000, help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of threads')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device index in form `cuda:N` (or `cpu`)')
    parser.add_argument('--checkpoint_dir', type=str, default='./re_fine_tuning', help='Directory for checkpoints')
    return parser


def get_sample_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models sampler script', description='available models')
    add_sample_args(subparsers.add_parser(MODEL.get_model_names()))
    return parser

def get_ft_model_path(config):
    return os.path.join(config.checkpoint_dir, 're_ft_model.pt')

# def get_ft_log_path(config):
#     return os.path.join(config.checkpoint_dir, 'pretrained_log.txt')

def get_ft_config_path(config):
    return os.path.join(config.checkpoint_dir, 're_ft_config.pt')

def get_ft_vocab_path(config):
    return os.path.join(config.checkpoint_dir, 're_ft_vocab.pt')

def get_generation_path(config):
    return os.path.join(config.checkpoint_dir, 'generated_chemicals_100w.csv')

parser = get_common_parser()
common_config = parser.parse_known_args()[0]
if not os.path.exists(common_config.checkpoint_dir):
    os.mkdir(common_config.checkpoint_dir)

MODEL = ModelStorage()
model_name = MODEL.get_model_names()
model_path = get_ft_model_path(common_config)
config_path = get_ft_config_path(common_config)
vocab_path = get_ft_vocab_path(common_config)
# log_path = get_ft_log_path(common_config)

assert os.path.exists(model_path), ("Can't find model path for sampling: '{}'".format(model_path))
assert os.path.exists(config_path), ("Can't find config path for sampling: '{}'".format(config_path))
assert os.path.exists(vocab_path), ("Can't find vocab path for sampling: '{}'".format(vocab_path))

sampler_parser = get_sample_parser()
sampler_args = ['--device', common_config.device,
                '--model_load', model_path,
                '--config_load', config_path,
                '--vocab_load', vocab_path,
                '--gen_save', get_generation_path(common_config),
                '--n_samples', str(common_config.n_samples)]
sampler_config = sampler_parser.parse_known_args([model_name] + sys.argv[1:] +sampler_args)[0]
# sampler_script.main(model, sampler_config)
set_seed(sampler_config.seed)
device = torch.device(common_config.device)

# For CUDNN to work properly:
torch.cuda.set_device(device.index or 0)

model_config = torch.load(sampler_config.config_load)
model_vocab = torch.load(sampler_config.vocab_load)
model_state = torch.load(sampler_config.model_load)

model = MODEL.get_model_class(model_name)(model_vocab, model_config)
model.load_state_dict(model_state, strict=False)
model = model.to(device)
model.eval()

samples = []
n = sampler_config.n_samples
with tqdm(total=sampler_config.n_samples, desc='Generating samples') as T:
    while n > 0:
        current_samples = model.sample(min(n, sampler_config.n_batch), sampler_config.max_len)
        samples.extend(current_samples)

        n -= len(current_samples)
        T.update(len(current_samples))

samples = pd.DataFrame(samples, columns=['SMILES'])
samples.to_csv(sampler_config.gen_save, index=False)

