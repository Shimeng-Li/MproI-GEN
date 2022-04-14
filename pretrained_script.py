import os
import argparse
import sys
from model_storage import ModelStorage
import torch
from utils import add_train_args, set_seed, read_smiles_csv


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../../Dataset/PretrainedData/pretrained_zinc_500w/pretrained_zinc_500w_v3/train.csv',
                        required=False, help='Path to train molecules csv')
    parser.add_argument('--test_path', type=str, default='../../Dataset/PretrainedData/pretrained_zinc_500w/pretrained_zinc_500w_v3/test.csv',
                        required=False, help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path', default='../../Dataset/PretrainedData/pretrained_zinc_500w/pretrained_zinc_500w_v3/test_scaffolds.csv',
                        type=str, required=False, help='Path to scaffold test molecules csv')
    parser.add_argument('--checkpoint_dir', type=str, default='./pretrained_dir',
                        help='Directory for checkpoints')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of threads')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device index in form `cuda:N` (or `cpu`)')
    return parser


def get_trainer_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models Training', description='available models')
    add_train_args(
        MODEL.get_model_train_parser(MODEL.get_model_names())(subparsers.add_parser(MODEL.get_model_names())))
    return parser


def get_model_path(config):
    return os.path.join(config.checkpoint_dir, 'pretrained_model.pt')


def get_log_path(config):
    return os.path.join(config.checkpoint_dir, 'pretrained_log.txt')


def get_config_path(config):
    return os.path.join(config.checkpoint_dir, 'pretrained_config.pt')


def get_vocab_path(config):
    return os.path.join(config.checkpoint_dir, 'pretrained_vocab.pt')

def get_loss_path(config):
    return os.path.join(config.checkpoint_dir, 'loss')

def get_running_loss_path(config):
    return os.path.join(config.checkpoint_dir, 'running_loss')


MODEL = ModelStorage()
pretrained_parser = get_common_parser()
config = pretrained_parser.parse_known_args()[0]
if not os.path.exists(config.checkpoint_dir):
    os.mkdir(config.checkpoint_dir)

train_path = config.train_path
test_path = config.test_path
test_scaffolds_path = config.test_scaffolds_path

model_name = MODEL.get_model_names()
# train_model(config, model_name, train_path, test_path)
# def train_model(config, train_path, test_path):
model_path = get_model_path(config)
config_path = get_config_path(config)
vocab_path = get_vocab_path(config)
log_path = get_log_path(config)
loss_path = get_loss_path(config)
running_loss_path = get_running_loss_path(config)

trainer_parser = get_trainer_parser()
train_args = ['--device', config.device,
              '--model_save', model_path,
              '--config_save', config_path,
              '--vocab_save', vocab_path,
              '--log_file', log_path,
              '--n_jobs', str(config.n_jobs),
              '--loss_save', loss_path,
              '--running_loss_save', running_loss_path]
train_args.extend(['--train_load', train_path])
train_args.extend(['--val_load', test_path])

trainer_config = trainer_parser.parse_known_args([model_name] + sys.argv[1:] + train_args)[0]
# trainer_script.main(model, trainer_config)

set_seed(trainer_config.seed)
device = torch.device(trainer_config.device)
torch.save(trainer_config, trainer_config.config_save)

# For CUDNN to work properly
torch.cuda.set_device(device.index)

train_data = read_smiles_csv(trainer_config.train_load)
val_data = read_smiles_csv(trainer_config.val_load)
trainer = MODEL.get_model_trainer(model_name)(trainer_config)

vocab = trainer.get_vocabulary(train_data)
torch.save(vocab, trainer_config.vocab_save)

model = MODEL.get_model_class(model_name)(vocab, trainer_config).to(device)
trainer.fit(model, train_data, val_data)
model = model.to('cpu')
torch.save(model.state_dict(), trainer_config.model_save)
