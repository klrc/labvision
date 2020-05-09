from .core import AutoCore
from ._config import Config
from .. import auto


def init(project: str):
    print('init labvision.auto ..')
    auto.config = Config()
    print(f'init labvision.auto.config ..')
    auto.config.project = project
    print(f'init project as {project}')


def compile():
    args = auto.config.compile()
    return AutoCore(**args)
