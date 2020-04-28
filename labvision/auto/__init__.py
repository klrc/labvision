
from .evolution import Evolution

seed = 1
config = None
default_config = {}


def compile(config=None):
    raise NotImplementedError


def check_config():
    raise NotImplementedError
