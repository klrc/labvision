from . import backend
from .dock import set_server, set_user, deploy, run, close


class ServerConfig():
    def __init__(self):
        self.reset()

    def reset(self):
        self.local_root = None
        self.remote_root = None

    def cache_dir(self):
        return f'{self.local_root}/__cache__'


config = ServerConfig()


__all__ = ['backend', 'set_server', 'set_user', 'deploy', 'run', 'close', 'config']


def init(local_root, remote_root,):
    config.reset()
    print('init labvision.server ..')
    config.local_root = local_root
    config.remote_root = remote_root
