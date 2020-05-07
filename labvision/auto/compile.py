import functools
import inspect
import random
import torch
import labvision
from .utils import MissingArgsException, ExceptionMessage
from .controller import AutoControl


def printc(msg):
    print(f'\t[compiler] {msg}')


def check_input_isinstance(compile_type='default'):
    if compile_type is None:
        Args = _CompileArgs
    elif compile_type == 'optimizer':
        Args = _OptimizerArgs

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, x):
            if inspect.isclass(x):
                x = Args(x)
            return func(self, x)
        return wrapper
    return decorator


class _CompileArgs():
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
                printc(f'set {k} = {v}')
                assert v is not inspect._empty
        except Exception:
            print(args)
            raise MissingArgsException(k, 'This value must be specified.')
        return self.compile_class(**args)


class _OptimizerArgs(_CompileArgs):
    def __init__(self, compile_class):
        super().__init__(compile_class)
        self.lr = 1e-3
        self.weight_decay = 5e-4


class Template():
    class _Datasets():
        def __init__(self):
            self.trainset = None
            self.testset = None
            self.valset = None
            self.batch_size = 64
            self.num_workers = 1

        def compile(self):
            reserved = ['trainset', 'testset', 'valset', 'batch_size', 'num_workers']
            for k, v in self.__dict__.items():
                if k not in reserved:
                    self.trainset.__dict__[k] = v
                    self.testset.__dict__[k] = v
                    self.valset.__dict__[k] = v
            if isinstance(self.trainset, _CompileArgs):
                printc(f'compiling trainset: {self.trainset.compile_class} (train = True)')
                self.trainset = self.trainset.compile(train=True)
            if isinstance(self.testset, _CompileArgs):
                printc(f'compiling testset: {self.testset.compile_class} (train = False)')
                self.testset = self.testset.compile(train=False)
            if isinstance(self.valset, _CompileArgs):
                printc(f'compiling valset: {self.valset.compile_class} (train = False)')
                self.valset = self.valset.compile(train=False)
            self.num_workers = min(self.num_workers, labvision.io.backends.device.cpu_num_workers_limit)
            printc(f'set num_workers = {self.num_workers}')
            return self

    def __init__(self):
        self.seed = None
        self.__model = None
        self.__optimizer = None
        self.__datasets = Template._Datasets()
        self.status = None
        self.build_dir = './build'
        self.log_path = './rec.log'

    def compile(self):
        print('\n\t[compiler] compiling labvision.auto configs..')
        if self.seed is None:
            random.seed()
            self.seed = random.randint(0, 1e8)
            printc(f'initialize seed = {self.seed}')
        assert self.__model is not None
        if self.__optimizer is None:
            self.__optimizer = _OptimizerArgs(torch.optim.SGD)
        if isinstance(self.__optimizer, _CompileArgs):
            printc(f'compiling optimizer from {self.__optimizer.compile_class}')
            self.__optimizer = self.__optimizer.compile(params=self.__model.parameters())
        printc(f'compiling datasets ..')
        self.__datasets = self.__datasets.compile()
        printc('compile finished.\n')
        return self

    def freeze(self):
        return self.__dict__

    def melt(self, _dict):
        self.__dict__ = _dict
        return self

    @property
    def model(self):
        return self.__model

    @model.setter
    @check_input_isinstance('default')
    def model(self, model):
        self.__model = model

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    @check_input_isinstance('optimizer')
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def datasets(self):
        return self.__datasets

    @datasets.setter
    def datasets(self, datasets):
        if not inspect.isclass(datasets):
            raise ExceptionMessage('you should init datasets.trainset & datasets.testset(at least) seperately.')
        self.__datasets.trainset = _CompileArgs(datasets)
        self.__datasets.testset = _CompileArgs(datasets)
        self.__datasets.valset = _CompileArgs(datasets)



def get_default_config():
    return Template()


def compile(config):
    return AutoControl(config.compile())


def resume(fp):
    config = Template().melt(torch.load(fp))
    ac = AutoControl(config)
    ac.__log__(f'continue from: {fp}', pure_log=True)
    return ac
