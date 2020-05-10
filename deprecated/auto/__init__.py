from . import utils
from . import functional
from .core import AutoCore


__all__ = ['utils', 'functional', 'AutoCore']


config = utils.Config()


def init(project: str):
    global config
    print('init labvision.auto ..')
    config.reset()
    print(f'init labvision.auto.config ..')
    config.project = project
    print(f'init project as {project}')


def compile():
    global config
    args = config.compile()
    return AutoCore(**args)
