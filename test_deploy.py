
from labvision import dock
server = dock.Server('/home/sh/Desktop/Research/remote', ip='172.20.46.235', user='sh', password='sh')
with server:
    server.deploy('test-project', 'test.py', conda_env='dl')
