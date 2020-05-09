from . import auto
from . import server
from . import utils
from . import datasets
from . import visualize
from . import transforms

__all__ = ['auto', 'server', 'utils', 'datasets', 'visualize', 'transforms']


def init(project: str):
    utils.init(project=project)
    auto.init()
    server.init()
