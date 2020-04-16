import torch
import os
import random
import string
import time
from torch.autograd import Variable
import json


class Status():
    def __init__(self):
        self.status_dict = {'hash': self.__genhash__()}

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
        return self.status_dict['train']['epoch']

    def iter(self):
        return self.status_dict['train']['iter']

    def epoch_finished(self):
        return self.status_dict['train']['epoch_finished']

    def freeze(self, fp):
        with open(fp, 'w') as f:
            json.dump(self.status_dict, f)

    def load(self, fp):
        with open(fp, 'r') as f:
            status_dict = json.load(f)
            self.update(status_dict)

    def hash(self):
        return self.status_dict['hash']

    def optimizer(self):
        return self.status_dict['optimizer']


class AutoControl():

    def __init__(self, dataset, model,
                 dataset_transforms=None, dataset_args={},
                 optimizer=torch.optim.SGD, optimizer_args={},
                 forward_func=None, loss_func=None, eval_func=None, **fpargs):
        self._init_datasets(dataset, dataset_transforms, **dataset_args)
        self.model = model
        self._init_optimizer(optimizer, **optimizer_args)
        self._forward_func = forward_func if forward_func else self.__forward__
        self._loss_func = loss_func if loss_func else self.__loss__
        self._eval_func = eval_func if eval_func else self.__eval__
        self.fps = self._check_fpargs(**fpargs)
        self.status = Status()
        self.under_test = False

    def load_status(self, fp, recover_optimizer=False):
        self.status.load(fp)
        if recover_optimizer:
            self.optimizer.load_state_dict(self.status.optimizer())
        return self

    def save_status(self, fp):
        self.status.freeze(fp)
        return self

    def load_model(self, fp):
        state_dict = torch.load(fp)
        self.model.load_state_dict(state_dict)
        return self

    def _check_fpargs(self, log_fp='external/rec.log', **args):
        fp_list = {
            'log_fp': log_fp,
        }
        for fp in fp_list:
            assert os.path.exists(fp)
        return fp_list

    def _init_datasets(self, dataset, dataset_transforms, batch_size=32, num_workers=2, **args):
        """
            Args:
                dataset:
                dataset_transforms: transform (for all) or Tuple(transform_train, transform_test)
                batch_size:
                num_workers:

                **args: args for Dataset.__init__().
        """
        if type(dataset_transforms) is tuple:
            _transform_train, _transform_test = dataset_transforms
        else:
            _transform_train = _transform_test = dataset_transforms
        trainset = dataset(train=True, transform=_transform_train, **args)
        testset = dataset(train=False, transform=_transform_test, **args)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def _init_optimizer(self, optimizer, **args):
        if 'params' not in args:
            args['params'] = self.model.parameters()
        if 'lr' not in args:
            args['lr'] = 1e-3
        self.optimizer = optimizer(**args)

    def log(self, msg, display=True):
        """
            Args:
                msg: str or metrics dict.

            example:
                >>> ac.log('acc@top1:'+acc)
                >>> ac.log(metrics)
        """
        if type(msg) is dict:
            for k in msg:
                self.__log__(f'[{self.status.epoch()}, {self.status.iter():5d}/{len(self.trainloader)}] {k}: {msg[k]}', display=display)
            return self
        return self.__log__(f'[{self.status.epoch()}, {self.status.iter():5d}/{len(self.trainloader)}] {msg}', display=display)

    def __log__(self, msg, display=True):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]<{self.status.hash()}>\t'+msg
        if self.under_test:
            line = f'testing# {line}'
        open(self.fps['log_fp'], 'a').write(line+'\n')
        if display:
            print(line)
        return self

    def check(self):
        self.under_test = True
        self.__log__('testing ..')
        self.__log__(self.__str__())
        _ = self.step(interval=1, val_interval=1)
        self.__log__('test passed.')
        self.under_test = False

    def eval(self, auto_log=True):
        """
            evaluate model accuracy,
            Args:
                auto_log:
        """
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for item in self.testloader:
                batch_metrics = self._eval_func(self.model, item)
                for key in batch_metrics:
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += batch_metrics[key]/len(self.testloader)
                if self.under_test:
                    break
        if auto_log:
            self.log(metrics)
        return metrics

    @staticmethod
    def __eval__(model, item, ks=(1, 3)):
        x, targets = item
        logits = model(Variable(x).cuda())
        metrics = {f'acc@top{k}': 0 for k in ks}
        for k in ks:
            metrics[f'acc@top{k}'] += (logits.topk(max((1, k)), 1, True, True)[1] == targets.view(-1, 1)).sum().float().item()/logits.shape[0]

    def step(self, interval=1, val_interval=4, auto_log=True):
        """
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
            status_optimizer = self.optimizer.state_dict()
            status_dict = {'train': status_train, 'val': status_val, 'optimizer': status_optimizer}
            self.status.update(status_dict)
            if auto_log:
                self.log(f'train_loss: {status_train["loss"]}')
                self.log(f'val_loss: {status_val["loss"]}')
            yield self.model, self.status

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
            for i, item in enumerate(self.trainloader, 0):
                loss += self.__step__(item, train=True)
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
            for i, item in enumerate(self.valloader, 0):
                loss += self.__step__(item, train=False)
                if (i+1) % yield_batches == 0:
                    loss /= yield_batches
                    yield {
                        'loss': loss,
                    }
                    loss = 0.0

    def __step__(self, item, train=True):
        """
            Args:
                item: a sample from dataloader.
                train: train=True if step for train.
        """
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            logits = self._forward_func(self.model, item)
            loss = self._loss_func(logits, item)
            loss.backward()
            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self._forward_func(self.model, item)
        return loss.item()

    @staticmethod
    def __forward__(model, item):
        x, _ = item
        return model(Variable(x).cuda())

    @staticmethod
    def __loss__(logits, item):
        _, targets = item
        return torch.nn.functional.cross_entropy(logits, Variable(targets).cuda())


if __name__ == "__main__":
    pass
