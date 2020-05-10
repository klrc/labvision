import random
import torch
import torch.nn as nn
import os
import inspect
import functools


class ExceptionMessage(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return f'{self.msg}'


class MissingArgsException(ExceptionMessage):
    def __init__(self, key, msg):
        msg = f'Missing key \'{key}\' in config\n({msg})'
        super().__init__(msg)


def check_input_isinstance(compile_type='default'):
    if compile_type is None:
        hint_type = Args
    elif compile_type == 'datasets':
        hint_type = DatasetsHint
    elif compile_type == 'optimizer':
        hint_type = OptimizerHint

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, x):
            if inspect.isclass(x):
                x = hint_type(x)
            return func(self, x)
        return wrapper
    return decorator


class Args():
    def __init__(self, compile_class):
        parameters = inspect.signature(compile_class).parameters

        def default_args(v):
            if v.default is not inspect._empty:
                return v.default
            if v.kind == inspect.Parameter.VAR_KEYWORD:
                return {}
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                return []

        parameter_dict = {k: default_args(v) for k, v in parameters.items()}
        self.compile_class = compile_class
        self.__dict__.update(parameter_dict)

    def __str__(self):
        return str(self.__dict__)

    def compile(self, **kwargs):
        args = {k: v for k, v in self.__dict__.items() if k != 'compile_class' and k != 'kwargs'}
        args.update(kwargs)
        try:
            for k, v in args.items():
                print(f'set {k} = {v}')
                assert v is not inspect._empty
        except Exception:
            raise MissingArgsException(k, 'This value must be specified.')
        return self.compile_class(**args)


class OptimizerHint(Args):
    def __init__(self, compile_class):
        super().__init__(compile_class)
        self.lr = 1e-3
        self.weight_decay = 5e-4


class DatasetsHint(Args):
    def __init__(self, compile_class):
        super().__init__(compile_class)
        self.root = None
        self.transform = None
        self.batch_size = 64
        self.num_workers = 1

    def compile(self, **kwargs):
        datasets = Datasets()
        datasets.trainset = super().compile(train=True)
        datasets.testset = super().compile(train=False)
        datasets.batch_size = self.batch_size
        datasets.num_workers = self.num_workers
        return datasets.compile()


class Datasets():
    def __init__(self):
        self.trainset = None
        self.testset = None
        self.valset = None
        self.batch_size = 64
        self.num_workers = 1

    def compile(self):
        assert self.trainset is not None
        assert self.testset is not None
        if self.valset is None:
            self.valset = self.testset
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return trainloader, valloader, testloader


class Config():
    def __init__(self):
        self.reset()

    def reset(self):
        self.seed = None
        self.model = None
        self.cuda = True
        self.__project = None
        self.__optimizer = None
        self.__datasets = Datasets()

    @property
    def datasets(self):
        return self.__datasets

    @datasets.setter
    @check_input_isinstance('datasets')
    def datasets(self, datasets):
        self.__datasets = datasets

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    @check_input_isinstance('optimizer')
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def project(self):
        return self.__project

    @project.setter
    def project(self, fp: str):
        if not os.path.exists(fp):
            print(f'path not exists, creating dir: {fp}')
            os.makedirs(fp)
        self.__project = fp

    def compile(self):
        print('compiling labvision.auto configs..')
        root = self.__project
        seed = self.seed
        model = self.model
        optimizer = self.__optimizer
        datasets = self.__datasets
        if seed is None:
            random.seed()
            seed = random.randint(0, 1e8)
        print(f'initialize seed = {seed}')
        assert model is not None
        if self.cuda:
            model = self.model.cuda()
        if optimizer is None:
            optimizer = OptimizerHint(torch.optim.SGD)
        if isinstance(optimizer, OptimizerHint):
            print(f'compiling optimizer from {optimizer.compile_class}')
            optimizer = optimizer.compile(params=model.parameters())
        print(f'compiling datasets ..')
        trainloader, valloader, testloader = datasets.compile()
        compiled_args = dict(
            root=root,
            seed=seed,
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
        )
        argdict = {k.replace('_Config__', ''): v for k, v in self.__dict__.items()}
        argdict = {k: v for k, v in argdict.items() if k not in ['project', 'datasets'] and k not in compiled_args.keys()}
        compiled_args.update(argdict)
        print('compile finished.\n')
        return compiled_args

    def __str__(self):
        ret = 'config:\n'
        for k, v in self.__dict__.items():
            str_v = v
            if type(v) is str:
                str_v = f'\"{v}\"'
            if isinstance(v, nn.Module):
                str_v = v._get_name()
            ret += f"    {k.replace('_Config__','')}: {str_v}\n"
        return ret
