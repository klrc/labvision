from . import _config
from . import io
from . import exceptions
__all__ = ['_config', 'io', 'exceptions']


from labvision import auto


def init(project: str):
    auto.config = _config.Config()
    print(f'init labvision.utils.config ..')
    auto.config.project = project
    print(f'init project as {project}')
