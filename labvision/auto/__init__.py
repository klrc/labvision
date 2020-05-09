from . import core
from . import functional
from .core import AutoCore
__all__ = ['core', 'functional', 'AutoCore']

config = None


def init():
    print('init labvision.auto ..')


def compile():
    global config
    args = config.compile()
    return core.AutoCore(**args)
