import time
import tarfile
import os
from .utils import check_dir
from .backends import ssh


cache_dir = 'build/__cache__'


def set_server(ip, target_location, port=22):
    ssh.set_server(ip, target_location, port)


def set_user(username, password):
    ssh.set_user(username, password)


def pack(source_dir, cache_dir):
    source_files = [x for x in os.listdir(source_dir)]
    assert 'main.py' in source_files
    check_dir(cache_dir)
    target_fp = f'{cache_dir}/deploy_pack.tar.gz'
    with tarfile.open(target_fp, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            for fname in files:
                pathfile = f'{root}/{fname}'
                arcpath = f'{root.split(source_dir)[-1]}/{fname}'
                tar.add(pathfile, arcname=arcpath)
    print(f'packing {target_fp}')
    return target_fp


def deploy(src='src', run='main.py', conda_env=None):
    global cache_dir
    deploy_pack = pack(src, cache_dir)
    fsize = os.path.getsize(deploy_pack)
    f_mb = fsize/float(1024)/float(1024)
    print(f'package size: {f_mb*1024:.6f}KB ({f_mb:.4f}MB)')
    if f_mb > 10:
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
