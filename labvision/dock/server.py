
import os
import re
import time

import paramiko

from .local import pack


class SSHConnector():
    def __init__(self, ip, user, password, port=22):
        self.ssh = None
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.cmd_location = '~'

    def __str__(self):
        return f'{self.user}@{self.ip}:{self.port}'

    def log(self, msg, **kwargs):
        print(f'\t[{self.ip}] {msg}', **kwargs)

    def __enter__(self):
        if self.ssh is None:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def connect(self):
        ssh = paramiko.SSHClient()  # 创建sshclient
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 指定当对方主机没有本机公钥的情况时应该怎么办，AutoAddPolicy表示自动在对方主机保存下本机的秘钥
        ssh.connect(self.ip, self.port, self.user, self.password)
        self.log(f'connected to {self.__str__()}')
        self.ssh = ssh
        return self

    def close(self):
        self.log('ssh closed.')
        self.ssh.close()
        self.ssh = None

    def cd(self, location):
        if self.ssh is None:
            self.connect()
        self.log(f'server:{self.cmd_location}$ cd {location}')
        _, stdout, stderr = self.ssh.exec_command(f'cd {location};pwd',)
        self.cmd_location = stdout.readlines()[0].replace('\n', '')
        for line in stderr:
            self.log(f'{self.cmd_location}$ {line}', end='')
        return stdout, stderr

    def __call__(self, command):
        return self.exec_command(command)

    def exec_command(self, command):
        if self.ssh is None:
            self.connect()
        if command.startswith('cd'):
            return self.cd(command[3:])
        self.log(f'server:{self.cmd_location}$ {command}')
        _, stdout, stderr = self.ssh.exec_command(f'. /etc/profile;. .bashrc;cd {self.cmd_location};{command}')
        out = stdout.readlines()
        for line in out:
            self.log(f'server:{self.cmd_location}$ {line}', end='')
        err = stderr.readlines()
        for line in err:
            self.log(f'server:{self.cmd_location}$ {line}', end='')
        return out, err

    @staticmethod
    def sftp_callback(transferred, total):
        print(f'\t[sftp] transferred/total: {transferred}/{total} ({transferred/total:%})')

    def upload(self, local_file, remote_file):
        if self.ssh is None:
            self.connect()
        _dir = re.sub(remote_file.split('/')[-1], '', remote_file)
        self.exec_command(f'mkdir {_dir}')
        sftp = self.ssh.open_sftp()
        sftp.put(local_file, remote_file, callback=self.sftp_callback)


class Server():
    def __init__(self, remote_root, **kwargs):
        self.ssh = SSHConnector(**kwargs)
        self.cache_dir = 'build'
        self.remote_root = remote_root
        self.fsize_limit = 1024

    def deploy(self, src, run='main.py', conda_env=None):
        deploy_pack = pack(src, self.cache_dir)
        fsize = os.path.getsize(deploy_pack)
        f_mb = fsize/float(1024)/float(1024)
        print(f'package size: {f_mb*1024:.6f}KB ({f_mb:.4f}MB)')
        if f_mb > self.fsize_limit:
            print('src too large for ssh.')
            raise NotImplementedError
        print('transfering deploy package to remote ssh ..')
        self.ssh.log('cleanning target dir..')
        self.ssh.exec_command(f'rm -r {self.remote_root}')
        self.ssh.log('sftp connection established, uploading ..')
        self.ssh.upload(deploy_pack, f'{self.remote_root}/deploy_pack.tar.gz')
        self.ssh.exec_command(f'cd {self.remote_root}')
        self.ssh.exec_command(f'tar -zxvf deploy_pack.tar.gz')
        if conda_env:
            self.ssh.exec_command(f'source activate {conda_env};nohup python {run} >log.d 2>&1 &')
        else:
            self.ssh.exec_command(f'screen -s labvision-remote;nohup python {run} >log.d 2>&1 &')
        self.ssh.log(f'successfully deployed, running in background now.')
        self.ssh.log(f'tracking python process:')
        self.ssh.exec_command(f'ps -u {self.ssh.user}|grep python')
        self.ssh.log(f"you can use 'cd {self.remote_root};cat log.d' to see the log.")
        print(f'removing cached deploy package: {deploy_pack}')
        os.remove(deploy_pack)

    def exec_command(self, command):
        return self.ssh(command)

    def __call__(self, command):
        return self.ssh(command)

    def close(self, latency=1):
        self.ssh.log(f'ssh close in {latency} secs ..')
        time.sleep(latency)
        self.ssh.close()

    def __enter__(self):
        self.ssh.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
