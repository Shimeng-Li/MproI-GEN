import os
import argparse
import sys
from model_storage import ModelStorage
import torch
from utils import add_fine_tuning_args, set_seed, read_smiles_csv

def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_train_path', type=str, default='../../Dataset/FineTuningData/sars-cov-2/pos_data.csv', help='Path to train molecules csv')
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_dir/pretrained_model.pt', help='Directory for pretrained model')
    parser.add_argument('--pretrained_vocab_path', type=str, default='./pretrained_dir/pretrained_vocab.pt', help='Directory for pretrained model')
    parser.add_argument('--pretrained_config_path', type=str, default='./pretrained_dir/pretrained_config.pt', help='Directory for pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default='./fine_tuning_for_tutorial', help='Directory for checkpoints')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of threads')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device index in form `cuda:N` (or `cpu`)')
    return parser


def get_fine_tuning_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models Fine Tuning', description='available models')
    add_fine_tuning_args(
        MODEL.get_model_train_parser(MODEL.get_model_names())(subparsers.add_parser(MODEL.get_model_names())))
    return parser


def get_ft_model_path(config):
    return os.path.join(config.checkpoint_dir, 'ft_model.pt')


def get_ft_log_path(config):
    return os.path.join(config.checkpoint_dir, 'ft_log.txt')


def get_ft_config_path(config):
    return os.path.join(config.checkpoint_dir, 'ft_config.pt')


def get_ft_vocab_path(config):
    return os.path.join(config.checkpoint_dir, 'ft_vocab.pt')


parser = get_common_parser()
common_config = parser.parse_known_args()[0]
if not os.path.exists(common_config.checkpoint_dir):
    os.mkdir(common_config.checkpoint_dir)

MODEL = ModelStorage()
model_name = MODEL.get_model_names()
ft_train_path = common_config.ft_train_path
pretrained_model_path = common_config.pretrained_model_path
pretrained_vocab_path = common_config.pretrained_vocab_path
pretrained_config_path = common_config.pretrained_config_path
ft_model_path = get_ft_model_path(common_config)
ft_config_path = get_ft_config_path(common_config)
ft_vocab_path = get_ft_vocab_path(common_config)
ft_log_path = get_ft_log_path(common_config)

ft_parser = get_fine_tuning_parser()
ft_args = ['--device', common_config.device,
           '--ft_model_save', ft_model_path,
           '--pretrained_model_load', pretrained_model_path,
           '--pretrained_config_load', pretrained_config_path,
           '--pretrained_vocab_load', pretrained_vocab_path,
           '--ft_config_save', ft_config_path,
           '--ft_vocab_save', ft_vocab_path,
           '--ft_log_file', ft_log_path,
           '--n_jobs', str(common_config.n_jobs)]

ft_config = ft_parser.parse_known_args([model_name] + sys.argv[1:] + ft_args)[0]

set_seed(ft_config.seed)
device = torch.device(ft_config.device)
torch.save(ft_config, ft_config.ft_config_save)

# For CUDNN to work properly:
torch.cuda.set_device(device.index or 0)
ft_train_data = read_smiles_csv(common_config.ft_train_path)
ft_trainer = MODEL.get_model_trainer(model_name)(ft_config)
ft_vocab = ft_trainer.get_vocabulary(ft_train_data)
torch.save(ft_vocab, ft_config.ft_vocab_save)

'''
load model
'''
pretrained_vocab = torch.load(common_config.pretrained_vocab_path)
pretrained_config = torch.load(common_config.pretrained_config_path)
pretrained_model_state = torch.load(ft_config.pretrained_model_load)
pretrained_model = MODEL.get_model_class(model_name)(pretrained_vocab, pretrained_config).to(device)
pretrained_model.load_state_dict(pretrained_model_state, strict=False)

'''
Fine Tuning Training
'''
ft_model = MODEL.get_model_class(model_name)(ft_vocab, ft_config).to(device)
num_fc = pretrained_model.linear_layer.in_features
num_out = len(ft_vocab)
ft_model.linear_layer = torch.nn.Linear(num_fc, num_out).to(ft_config.device)
for param in ft_model.linear_layer.parameters():
    param.requires_grad = True

ft_trainer.fine_tuning_fit(ft_model, ft_train_data)
ft_model = ft_model.to('cpu')
torch.save(ft_model.state_dict(), ft_config.ft_model_save)
