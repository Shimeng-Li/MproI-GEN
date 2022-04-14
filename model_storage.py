import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from model_definition import CharRNN
from model_trainer import CharRNNTrainer
from model_config import get_parser as char_rnn_parser


class ModelStorage():

    def __init__(self):
        self._models = {}
        self.add_model('char_rnn', CharRNN, CharRNNTrainer, char_rnn_parser)

    def add_model(self, name, class_, trainer_, parser_):
        self._models[name] = {'class': class_, 'trainer': trainer_, 'parser': parser_}

    def get_model_names(self):
        return list(self._models.keys())[0]

    def get_model_trainer(self, name):
        return self._models[name]['trainer']

    def get_model_class(self, name):
        return self._models[name]['class']

    def get_model_train_parser(self, name):
        return self._models[name]['parser']
