import time
import random
import string
import numpy as np
import torch

from torch.autograd import Variable
from . import functional
from . import utils


class Status():
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


class AutoCore():
    def __init__(self, config=None, **kwargs):
        if type(config) is utils.Config:
            config = config.compile()
        if type(config) is dict:
            config.update(kwargs)
        else:
            config = kwargs
        self.init(**config)

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

    def init(self, root, seed, model, optimizer, trainloader, valloader, testloader, **kwargs):
        self.root = root
        self.init_seed(seed)
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.__train_generator = None
        self.__val_generator = None
        self.step_iters = 200
        self.val_iters = 16
        self.status = Status()
        self.eval_hooks = []
        self.under_test = False

    def init_seed(self, seed):
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn
        # https://zhuanlan.zhihu.com/p/76472385
        # torch.backends.cudnn.benchmark = False

    def log(self, msg, head=True):
        if type(msg) is dict:
            for k, v in msg.items():
                self.log(f'{k}: {v}')
            return self
        if head:
            msg_head = f'[{self.status.epoch}, {self.status.iter:5d}/{len(self.trainloader)}]'
            msg = f'{msg_head} {msg}'
        else:
            msg = msg
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]<{self.status.hash}>\t{msg}'
        if self.under_test:
            line = 'testing# ' + line
        print(line)
        with open(f'{self.root}/slave.log', 'a') as f:
            f.write(f'{line}\n')
        return self

    def steps(self, iters=None, val_iters=16, max_epoch=None):
        '''
            Args:
                iters: set a checkpoint per <iter> iterations.
                val_iters: total validation batches for each checkpoint.
                max_epoch:
        '''
        if iters:
            self.step_iters = iters
        self.val_iters = val_iters
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
        if iters:
            self.step_iters = iters
        if val_iters:
            self.val_iters = val_iters

        if self.__train_generator is None:
            self.__train_generator = self.__train__(start_epoch=self.status.epoch)
            self.__val_generator = self.__val__()

        state_train = next(self.__train_generator)
        state_val = next(self.__val_generator)
        self.status.train.update(state_train)
        self.status.val.update(state_val)
        self.log(f'train_loss: {state_train["loss"]}')
        self.log(f'val_loss: {state_val["loss"]}')
        return self.status

    def check(self, iters=1):
        self.under_test = True
        self.log(f'starting check ..', head=False)
        self.step(iters=iters, val_iters=1)
        self.log(f'check passed.', head=False)
        self.under_test = False

    def eval(self, _type=None):
        self.model.eval()
        if _type == 'acc':
            self.eval_hooks = [('acc@top1', functional.accuracy)]
        if _type == 'acc@top3':
            self.eval_hooks = [('acc@top3', functional.accuracy_top3)]
        metrics = {k[0]: 0 for k in self.eval_hooks}
        with torch.no_grad():
            for sample in self.testloader:
                x, y = self.read_batch(sample)
                logits = self.forward(x)
                for name, hook_fn in self.eval_hooks:
                    metrics[name] += hook_fn(logits, y)/len(self.testloader)
        self.log(metrics)
        if _type is not None:
            return metrics[_type]
        return metrics

    def metrics_hook(self, name, hook_fn):
        '''
            Args:
                hook_fn: hook_fn(logits, y) -> metrics_value
        '''
        self.eval_hooks.append((name, hook_fn))

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
