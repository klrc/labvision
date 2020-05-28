import inspect


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
                        args[k] = v = v.compile(logger=logger)
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

