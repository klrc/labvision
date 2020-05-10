import time
import os
from .backend import ssh
from labvision.io import pack
import labvision.server.config as config


def set_server(ip, target_location, port=22):
    ssh.set_server(ip, target_location, port)


def set_user(username, password):
    ssh.set_user(username, password)


def deploy(src=None, run='main.py', conda_env=None):
    if src is None:
        src = config.local_root
    deploy_pack = pack(src, config.cache_dir())
    fsize = os.path.getsize(deploy_pack)
    f_mb = fsize/float(1024)/float(1024)
    print(f'package size: {f_mb*1024:.6f}KB ({f_mb:.4f}MB)')
    if f_mb > 40:
        print('src too large for ssh.')
        raise NotImplementedError
    try:
        assert ssh.server_ip is not None
    except Exception:
        print('server_ip not specified, use labvision.io.ssh.set_server() to init remote server.')
        return
    try:
        assert ssh.server_location is not None
    except Exception:
        print('target_location not specified, use labvision.io.ssh.set_server() to init remote server.')
        return
    try:
        assert ssh.user is not None or ssh.password is not None
    except Exception:
        print('user or password not specified, use labvision.io.ssh.set_user() to init remote server.')
        return
    print('transfering deploy package to remote ssh ..')
    print('\t[remote] cleanning target dir..')
    ssh.exec_command(f'rm -r {ssh.server_location}')
    print('\t[remote] uploading ..')
    ssh.upload(deploy_pack, f'{ssh.server_location}/deploy_pack.tar.gz')
    ssh.exec_command(f'cd {ssh.server_location}')
    ssh.exec_command(f'tar -zxvf deploy_pack.tar.gz')
    if conda_env:
        ssh.exec_command(f'source activate {conda_env};nohup python {run} >log.d 2>&1 &')
    else:
        ssh.exec_command(f'screen -s labvision-remote;nohup python {run} >log.d 2>&1 &')
    print(f'\t[remote] successfully deployed, running in background now. (pid={os.getpid()})')
    print(f"\t[remote] you can use 'cd {ssh.server_location};cat log.d' to see the log.")
    print(f'removing cached deploy package: {deploy_pack}')
    os.remove(deploy_pack)


def run(command):
    ssh.exec_command(command)


def close(latency=1):
    print(f'\t[remote] ssh close in {latency} secs ..')
    time.sleep(latency)
    ssh.close()
