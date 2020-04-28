import numpy as np
import json
import os
import random
import string
import time

import torch
from torch.autograd import Variable


class Status():
    def __init__(self, status_dict=None):
        """
            inner status class of AutoControl,
            supports reading /saving /loading.
        """
        if status_dict is None:
            status_dict = {'hash': self.__genhash__()}
        self.status_dict = status_dict

    @staticmethod
    def __genhash__():
        """
            generates a hash with ASCII letters and digits,
            always starts with a letter (for markdown usage).
        """
        _hash_head = ''.join(random.sample(string.ascii_letters, 1))
        _hash_body = ''.join(random.sample(string.ascii_letters+string.digits, 7))
        return _hash_head+_hash_body

    def update(self, status_dict):
        self.status_dict.update(status_dict)

    def epoch(self):
        if 'train' not in self.status_dict:
            return 0
        return self.status_dict['train']['epoch']

    def iter(self):
        return self.status_dict['train']['iter']

    def epoch_finished(self):
        return self.status_dict['train']['epoch_finished']

    def hash(self):
        return self.status_dict['hash']


class Evolution():
    def __init__(self, config):
        self.config = config
        self._compile(config)
        self.under_test = False

    def __train__(self, yield_batches, start_epoch=0):
        """
            train over epoches infinitely (not really),
            Args:
                yield_batches: batches to report loss.
                start_epoch: epoch to start from.
        """
        epoch = start_epoch-1
        while True:
            loss = 0.0
            epoch += 1
            for i, sample in enumerate(self.trainloader, 0):
                loss += self.__step__(sample, train=True)
                if (i+1) % yield_batches == 0:
                    loss /= yield_batches
                    epoch_finished = False
                elif i == len(self.trainloader)-1:
                    loss /= len(self.trainloader) % yield_batches
                    epoch_finished = True
                else:
                    continue
                status = {
                    'loss': loss,
                    'epoch': epoch,
                    'iter': i+1,
                    'epoch_finished': epoch_finished,
                }
                loss = 0.0
                yield status

    def __val__(self, yield_batches):
        """
            evaluate eval_loss for fixed batches.
            Args:
                yield_batches: batches used for a single eval.
        """
        while True:
            loss = 0.0
            for i, sample in enumerate(self.valloader, 0):
                loss += self.__step__(sample, train=False)
                if (i+1) % yield_batches == 0:
                    loss /= yield_batches
                    yield {
                        'loss': loss,
                    }
                    loss = 0.0

    def __step__(self, sample, train=True):
        """
            Args:
                sample: batched sample from dataloader.
                train: train=True if step for train.
        """
        x, y = self.__read__(sample)
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.forward(x)
            loss = self.loss_function(logits, y)
            loss.backward()
            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.forward(x)
                loss = self.loss_function(logits, y)
        return loss.item()

    def __log__(self, msg, display=True):
        """
            Args:
                msg:
                display:
        """
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]<{self.status.hash()}>\t'+msg
        if self.under_test:
            line = f'testing# {line}'
        open(self.config['log_file_path'], 'a').write(line+'\n')
        if display:
            print(line)
        return self

    @staticmethod
    def __read__(sample):
        """
            read x,y from sample,
            override if needed.
        """
        x, y = sample
        return Variable(x).cuda(), Variable(y).cuda()

    def _init_seed(self, seed):
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn

    def _init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _init_dataloader(self, trainset, testset, valset, batch_size=32, num_workers=2, **kwargs):
        """
            Args:
                trainset:
                testset:
                batch_size:
                num_workers:
        """
        if valset is None:
            valset = testset
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def _compile(self, config):
        random.seed()
        self.status = Status()  # gen hash before init seed
        self._init_seed(config['seed'])
        self.model = config['model']
        self._init_optimizer(config['optimizer'])
        self._init_dataloader(**config['datasets'])

    def step(self, interval=1, val_interval=4, auto_log=True):
        """
            step to the next checkpoint,
            works as a generator.

            Args:
                interval:
                val_interval:
                auto_log:
        """

        train_generator = self.__train__(yield_batches=interval, start_epoch=self.status.epoch())
        val_generator = self.__val__(yield_batches=val_interval)

        while True:
            status_train = next(train_generator)
            status_val = next(val_generator)
            status_dict = {'train': status_train, 'val': status_val}
            self.status.update(status_dict)
            if auto_log:
                self.log(f'train_loss: {status_train["loss"]}')
                self.log(f'val_loss: {status_val["loss"]}')
            yield self.model, self.status

    def log(self, msg, display=True):
        """
            display(optional) & save log.
            Args:
                msg: str or metrics dict.
                display:

            example:
                >>> ac.log('acc@top1:'+acc)
                >>> ac.log(metrics)
        """
        if type(msg) is dict:
            for k in msg:
                self.__log__(f'[{self.status.epoch()}, {self.status.iter():5d}/{len(self.trainloader)}] {k}: {msg[k]}', display=display)
            return self
        return self.__log__(f'[{self.status.epoch()}, {self.status.iter():5d}/{len(self.trainloader)}] {msg}', display=display)

    def forward(self, x):
        """
            Args:
                x:
            override if needed.
        """
        return self.model(x)

    def freeze(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @staticmethod
    def loss_function(logits, y):
        """
            Args:
                logits:
                y:

            override if needed.
        """
        return torch.nn.functional.cross_entropy(logits, y)
