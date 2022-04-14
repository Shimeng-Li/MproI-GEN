import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils import MetaTrainer
from utils import CharVocab, Logger
import numpy as np

class CharRNNTrainer(MetaTrainer):

    def __init__(self, config):
        self.config = config

    def _train_epoch(self, model, tqdm_data, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()
        postfix = {'loss': 0,
                   'running_loss': 0}
        for i, (prevs, nexts, lens) in enumerate(tqdm_data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to(model.device)
            outputs, _, _ = model(prevs, lens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            if optimizer is not None:
                optimizer.zero_grad()
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)
        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)
        device = model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.config.step_size, self.config.gamma)
        model.zero_grad()
        loss_list = []
        running_loss_list = []
        for epoch in range(self.config.train_epochs):
            scheduler.step()
            tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)
            loss_list.append(postfix['loss'])
            running_loss_list.append(postfix['running_loss'])
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            if (self.config.model_save is not None) and (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)
        loss_tensor = np.array(loss_list)
        running_loss_tensor = np.array(running_loss_list)
        np.save(self.config.loss_save, loss_tensor)
        np.save(self.config.running_loss_save, running_loss_tensor)


    def _ft_train(self, model, train_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)
        device = model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.ft_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.config.ft_step_size, self.config.ft_gamma)
        model.zero_grad()
        ft_loss_list = []
        ft_running_loss_list = []
        for epoch in range(self.config.ft_epoch):
            scheduler.step()
            tqdm_data = tqdm(train_loader, desc='Fine Tuning (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)
            ft_loss_list.append(postfix['loss'])
            ft_running_loss_list.append(postfix['running_loss'])
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            if (self.config.ft_model_save is not None) and (epoch % self.config.ft_save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(), self.config.ft_model_save[:-3]+'_{0:03d}.pt'.format(epoch))
                model = model.to(device)
        ft_loss_tensor = np.array(ft_loss_list)
        ft_running_loss_tensor = np.array(ft_running_loss_list)
        # np.save(self.config.ft_loss_save, ft_loss_tensor)
        # np.save(self.config.ft_running_loss_save, ft_running_loss_tensor)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]
            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long, device=device)
            return prevs, nexts, lens
        return collate

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None
        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)
        self._train(model, train_loader, val_loader, logger)
        return model

    def fine_tuning_fit(self, model, train_data, val_data=None):
        # logger = Logger() if self.config.ft_log_file is not None else None
        train_loader = self.get_ft_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_ft_dataloader(model, val_data, shuffle=False)
        self._ft_train(model, train_loader, val_loader)
        return model
