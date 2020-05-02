from labvision.io import remote

if __name__ == "__main__":
    remote.set_server('172.20.46.235', target_location='/home/sh/Desktop/remote_deploy')
    remote.set_user('sh', 'sh')
    remote.deploy('/home/sh/Desktop/Research/src', run='main.py', conda_env='dl')
    remote.close(latency=1)
