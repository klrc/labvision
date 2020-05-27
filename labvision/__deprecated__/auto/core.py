

import re
import inspect
import os
import random
import string
import time
import json

import torch
from labvision import functional
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .utils import manual_seed


class MissingArgsException(Exception):
    def __init__(self, key, msg):
        super().__init__()
        msg = f'Missing key \'{key}\' in config\n({msg})'

    def __str__(self):
        return f'{self.msg}'


class Dry():
    class Args():
        def __init__(self, compile_class):
            parameters = inspect.signature(compile_class).parameters
            parameter_dict = {k: self.__defaultargs__(v) for k, v in parameters.items()}
            self.compile_class = compile_class
            self.__dict__.update(parameter_dict)

        def update(self, **kwargs):
            self.__dict__.update(kwargs)

        @staticmethod
        def __defaultargs__(v):
            if v.default is not inspect._empty:
                return v.default
            if v.kind == inspect.Parameter.VAR_KEYWORD:
                return {}
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                return []

        def __str__(self):
            return str(self.__dict__)

        def log(self, msg, logger=None):
            if logger is None:
                print(msg)
            else:
                logger.log(msg)

        def compile(self, logger=None, **kwargs):
            args = {k: v for k, v in self.__dict__.items() if k != 'compile_class' and k != 'kwargs'}
            args.update(kwargs)
            try:
                self.log(f'==> Compiling: {self.compile_class}', logger=logger)
                for k, v in args.items():
                    if isinstance(v, Dry.Args):
                        v = v.compile(logger=logger)
                    self.log(f'  -> Running args hook: {k} = {v}', logger=logger)
                    assert v is not inspect._empty
            except Exception:
                raise MissingArgsException(k, 'This value must be specified.')
            return self.compile_class(**args)

    class Optimizer(Args):
        def __init__(self, compile_class):
            super().__init__(compile_class)
            self.lr = 1e-3
            self.weight_decay = 5e-4

    class Dataset(Args):
        def __init__(self, compile_class):
            super().__init__(compile_class)
            self.root = None
            self.transform = None


class Core():
    def __init__(self, root='.'):
        self.__model = None
        self.__optimizer = None
        self.__trainset = None
        self.__testset = None
        self.__valset = None
        self.seed = None
        self.root = root
        self.batch_size = 16
        self.num_workers = 2
        self.netdata_log_path = None
        self.slave_log_path = None
        self.auto_log = True

    @property
    def model(self): return self.__model
    @model.setter
    def model(self, x):
        if inspect.isclass(x):
            x = Dry.Args(x)
        self.__model = x

    @property
    def optimizer(self): return self.__optimizer
    @optimizer.setter
    def optimizer(self, x):
        if inspect.isclass(x):
            x = Dry.Optimizer(x)
        self.__optimizer = x

    @property
    def trainset(self): return self.__trainset
    @trainset.setter
    def trainset(self, x):
        if inspect.isclass(x):
            x = Dry.Dataset(x)
        self.__trainset = x

    @property
    def testset(self): return self.__testset
    @testset.setter
    def testset(self, x):
        if inspect.isclass(x):
            x = Dry.Dataset(x)
        self.__testset = x

    @property
    def valset(self): return self.__valset
    @valset.setter
    def valset(self, x):
        if inspect.isclass(x):
            x = Dry.Dataset(x)
        self.__valset = x

    def set_datasets(self, dataset_class, **kwargs):
        self.trainset = Dry.Dataset(dataset_class)
        self.trainset.update(**kwargs)
        self.testset = Dry.Dataset(dataset_class)
        self.testset.update(**kwargs)

    def push(self, name, value, dry=False, **kwargs):
        if dry and inspect.isclass(value):
            value = Dry.Args(value)
            value.update(**kwargs)
        self.__dict__[name] = value

    def compile(self):
        return Slave(self)

    def freeze(self, fp):
        _dir = re.sub(fp.split('/')[-1], '', fp)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        torch.save(self, fp)
        return fp


class Slave():
    class CurrentStatus():
        def __init__(self, frozen_dict=None):
            if frozen_dict:
                self.state_dict = frozen_dict
            else:
                self.state_dict = dict(
                    hash=self.__genhash__(),
                    train={},
                    val={},
                )

        @property
        def train(self):
            return self.state_dict['train']

        @property
        def val(self):
            return self.state_dict['val']

        @staticmethod
        def __genhash__():
            """
                generates a hash with ASCII letters and digits,
                always starts with a letter (for markdown usage).
            """
            random.seed()
            _hash_head = ''.join(random.sample(string.ascii_letters, 1))
            _hash_body = ''.join(random.sample(string.ascii_letters+string.digits, 7))
            return _hash_head+_hash_body

        def epoch_finished(self):
            return self.train['epoch_finished']

        @property
        def hash(self):
            return self.state_dict['hash']

        @property
        def iter(self):
            return self.train['iter']

        @property
        def epoch(self):
            if 'epoch' not in self.train:
                return 0
            return self.train['epoch']

    def __init__(self, core, cuda=True):
        self.core = core
        self.cuda = cuda
        self.status = self.CurrentStatus()
        self.step_iters = 200
        self.val_iters = 16
        self._eval_hooks = []
        self.__train_generator = None
        self.__val_generator = None
        self.under_test = False
        self.netdata_state = dict(hash=self.hash)
        self.compile(core)

    def compile(self, core: Core):
        self.root = core.root
        if not os.path.exists(core.root):
            os.makedirs(core.root)
            self.log(f'path not exists, creating dir: {core.root}')
        self.log('start compiling ..')
        assert isinstance(core.model, torch.nn.Module)
        assert core.trainset is not None
        assert core.testset is not None
        if core.seed is None:
            random.seed()
            self.seed = random.randint(0, 1e8)
            self.log(f'initialize seed = {self.seed}')
        else:
            self.seed = core.seed
        self.init_seed(self.seed)
        if isinstance(core.model, Dry.Args):
            self.log(f'compiling model from {core.model.compile_class} ..')
            self.model = core.model.compile(logger=self)
        else:
            self.model = core.model
        if self.cuda:
            self.log(f'transfer model to cuda device ..')
            self.model = self.model.cuda()
        self.optimizer = core.optimizer
        if self.optimizer is None:
            self.optimizer = Dry.Optimizer(torch.optim.SGD)
            self.log(f'using default optimizer as <torch.optim.SGD>')
        if isinstance(self.optimizer, Dry.Optimizer):
            self.log(f'compiling optimizer from {self.optimizer.compile_class} ..')
            self.optimizer = self.optimizer.compile(params=self.model.parameters(), logger=self)
        self.log(f'compiling datasets ..')
        trainset = core.trainset
        testset = core.testset
        valset = core.valset
        if valset is None:
            valset = testset
        if isinstance(trainset, Dry.Dataset):
            trainset = trainset.compile(train=True, logger=self)
        if isinstance(testset, Dry.Dataset):
            testset = testset.compile(train=False, logger=self)
        if isinstance(valset, Dry.Dataset):
            valset = valset.compile(train=False, logger=self)
        self.log(f'compiling dataloaders ..')
        self.trainloader = DataLoader(trainset, batch_size=core.batch_size, shuffle=True, num_workers=core.num_workers)
        self.testloader = DataLoader(testset, batch_size=core.batch_size, shuffle=False, num_workers=core.num_workers)
        self.valloader = DataLoader(valset, batch_size=core.batch_size, shuffle=True, num_workers=core.num_workers)
        self.log('compile finished.')

    def epoch_finished(self): return self.status.epoch_finished()
    @property
    def hash(self): return self.status.hash
    @property
    def iter(self): return self.status.iter
    @property
    def epoch(self): return self.status.epoch

    def __step__(self, sample, train=True):
        """
            Args:
                sample: batched sample from dataloader.
                train: train=True if step for train.
        """
        x, y = self.read_batch(sample)
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

    def __train__(self, start_epoch=0):
        """
            train over epoches infinitely (not really),
            Args:
                start_epoch: epoch to start from.
        """
        epoch = start_epoch-1
        while True:
            epoch += 1
            last_iter = 0
            loss = 0.0
            for i, sample in enumerate(self.trainloader, 0):
                loss += self.__step__(sample, train=True)
                epoch_finished = False
                if (i+1) == len(self.trainloader):
                    epoch_finished = True
                if epoch_finished or i-last_iter >= self.step_iters:
                    loss /= i-last_iter
                    yield dict(loss=loss, epoch=epoch, iter=i+1, epoch_finished=epoch_finished)
                    last_iter = i
                    loss = 0.0

    def __val__(self):
        """
            evaluate eval_loss for fixed batches.
        """
        last_iter = 0
        current_iter = 0
        while True:
            loss = 0.0
            for sample in self.valloader:
                loss += self.__step__(sample, train=False)
                current_iter += 1
                if current_iter-last_iter >= self.val_iters:
                    loss /= current_iter-last_iter
                    yield dict(loss=loss)
                    last_iter = current_iter
                    loss = 0.0

    def init_seed(self, seed):
        self.log(f'init with seed = {seed}')
        manual_seed(seed)

    def log(self, line, time_head=True):
        if not self.core.auto_log:
            return
        if self.under_test:
            line = f'testing# {line}'
        if time_head:
            line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {line}'
        print(line)
        fp = self.core.slave_log_path
        if fp is None:
            fp = f'{self.root}/slave.log'
        with open(fp, 'a') as f:
            f.write(f'{line}\n')

    def netdata_log(self, metrics_dict):
        if not self.core.auto_log:
            return
        if self.core.netdata_log_path is not None:
            self.netdata_state.update(metrics_dict)
            with open(self.core.netdata_log_path, "w")as f:
                json.dump(self.netdata_state, f)

    def hyper_log(self, msg, time_head=True, hash_head=True, iter_head=True):
        if not self.core.auto_log:
            return
        if type(msg) is dict:
            for k, v in msg.items():
                self.hyper_log(f'{k}: {v}', time_head=time_head, hash_head=hash_head, iter_head=iter_head)
            return self
        if iter_head:
            msg = f'[{self.status.epoch}, {self.status.iter:5d}/{len(self.trainloader)}] {msg}'
        if hash_head:
            msg = f'<{self.status.hash}>\t{msg}'
        self.log(msg, time_head=time_head)

    def __setinterval__(self, iters=None, val_iters=None):
        if iters:
            assert iters > 0
            if iters < 1:
                iters = int(len(self.trainloader)*iters)
            self.step_iters = iters
        if val_iters:
            assert val_iters > 0
            if val_iters < 1:
                val_iters = int(len(self.valloader)*iters)
            self.val_iters = val_iters

    def steps(self, iters=None, val_iters=None, max_epoch=None):
        '''
            Args:
                iters: set a checkpoint per <iter> iterations.
                val_iters: total validation batches for each checkpoint.
                max_epoch:
        '''
        self.__setinterval__(iters=iters, val_iters=val_iters)
        while True:
            yield self.step()
            if max_epoch is not None and self.status.epoch > max_epoch:
                break

    def step(self, iters=None, val_iters=None):
        """
            step to the next checkpoint, works as a generator.
            yields: status
            Args:
                iters:
                val_iters:
        """
        self.__setinterval__(iters=iters, val_iters=val_iters)
        if self.__train_generator is None:
            self.__train_generator = self.__train__(start_epoch=self.status.epoch)
            self.__val_generator = self.__val__()

        state_train = next(self.__train_generator)
        state_val = next(self.__val_generator)
        self.status.train.update(state_train)
        self.status.val.update(state_val)
        self.hyper_log(f'train_loss: {state_train["loss"]}\tval_loss: {state_val["loss"]}')
        self.netdata_log(dict(train_loss=state_train["loss"], val_loss=state_val["loss"]))
        return self.status

    def check(self, iters=1):
        self.under_test = True
        self.hyper_log(f':: starting check ..', iter_head=False)
        self.step(iters=iters, val_iters=1)
        self.__train_generator = None
        self.__val_generator = None
        self.hyper_log(f':: check passed.', iter_head=False)
        self.under_test = False

    def eval(self, _type=None):
        self.model.eval()
        if _type == 'acc@top1':
            self._eval_hooks = [('acc@top1', functional.accuracy)]
        if _type == 'acc@top3':
            self._eval_hooks = [('acc@top3', functional.accuracy_top3)]
        metrics = {k[0]: 0 for k in self._eval_hooks}
        with torch.no_grad():
            for sample in self.testloader:
                x, y = self.read_batch(sample)
                logits = self.forward(x)
                for name, hook_fn in self._eval_hooks:
                    metrics[name] += hook_fn(logits, y)/len(self.testloader)
        self.hyper_log(metrics)
        self.netdata_log(metrics)
        if _type is not None:
            return metrics[_type]
        return metrics

    def metrics_hook(self, name, hook_fn):
        '''
            Args:
                hook_fn: hook_fn(logits, y) -> metrics_value
        '''
        self._eval_hooks.append((name, hook_fn))

    def forward(self, x):
        """
            Args:
                x:
            override if needed.
        """
        return self.model(x)

    @staticmethod
    def loss_function(logits, y):
        """
            Args:
                logits:
                y:

            override if needed.
        """
        return torch.nn.functional.cross_entropy(logits, y)

    @staticmethod
    def read_batch(sample):
        """
            read x,y from sample,
            override if needed.
        """
        x, y = sample
        return Variable(x).cuda(), Variable(y).cuda()
