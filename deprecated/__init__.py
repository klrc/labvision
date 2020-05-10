from . import auto
from . import server
from . import supports
from labvision.supports import transforms
from labvision.supports import datasets
from labvision.supports import visualize
from labvision.supports import io

__all__ = ['auto', 'server', 'supports', 'transforms', 'datasets', 'visualize', 'io']


def init(project: str):
    auto.init(project=project)
