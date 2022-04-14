import argparse

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers")
    model_arg.add_argument("--hidden", type=int, default=768, help="Hidden size")
    model_arg.add_argument("--dropout", type=float, default=0.2, help="dropout between LSTM layers except for last")
    # Model Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=40, help='Number of epochs for model training')
    train_arg.add_argument('--device', type=str, default='cuda:0', help='GPU device index in form `cuda:N` (or `cpu`)')
    train_arg.add_argument('--n_batch', type=int, default=512, help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10, help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=20, help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1, help='Number of workers for DataLoaders')
    train_arg.add_argument('--ft_epoch', type=int, default=50, help='Number of epochs for model training')
    train_arg.add_argument('--ft_batch', type=int, default=1, help='Size of fine tuning batch')
    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
