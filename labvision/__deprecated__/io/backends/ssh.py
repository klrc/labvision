import paramiko


server_ip = None
server_port = None
server_location = ''
port = None
user = None
password = None
ssh = None
command_location = '~'


def set_server(ip, target_location, port=22):
    global server_ip, server_location, server_port
    server_ip = ip
    server_location = target_location
    server_port = port


def set_username(_str):
    global user
    user = _str


def set_password(_str):
    global password
    password = _str


def set_user(username, pwd):
    set_username(username)
    set_password(pwd)


def connect():
    global server_ip, server_port, user, password
    # SSH远程连接
    ssh = paramiko.SSHClient()  # 创建sshclient
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 指定当对方主机没有本机公钥的情况时应该怎么办，AutoAddPolicy表示自动在对方主机保存下本机的秘钥
    ssh.connect(server_ip, server_port, user, password)
    print(f'\t[remote] connected to {user}@{server_ip}:{server_port}')
    return ssh


def close():
    global ssh
    if ssh is not None:
        print('\t[remote] ssh closed.\n')
        ssh.close()


def cd(location):
    global ssh, command_location
    if ssh is None:
        ssh = connect()
    print(f'\t[remote] server:{command_location}$ cd {location}')
    _, stdout, stderr = ssh.exec_command(f'cd {location};pwd',)
    command_location = stdout.readlines()[0].replace('\n', '')
    return stdout, stderr


def exec_command(command):
    global ssh, command_location
    if ssh is None:
        ssh = connect()
    if command.startswith('cd'):
        return cd(command[3:])
    print(f'\t[remote] server:{command_location}$ {command}')
    _, stdout, stderr = ssh.exec_command(f'. /etc/profile;. .bashrc;cd {command_location};{command}')
    out = stdout.readlines()
    for line in out:
        print(f'\t[remote] server:{command_location}$ {line}', end='')
    err = stderr.readlines()
    for line in err:
        print(f'\t[remote] server:{command_location}$ {line}', end='')
    return out, err


def sftp_callback(transferred, total):
    print(f'\t[sftp] transferred/total: {transferred}/{total} ({transferred/total:%})')


def upload(local_file, remote_file):
    global ssh
    if ssh is None:
        ssh = connect()
    exec_command(f'mkdir {server_location}')
    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_file, callback=sftp_callback)


if __name__ == "__main__":
    set_server('172.20.46.235', '/home/sh/Desktop/remote_deploy')
    set_user('sh', 'sh')
    exec_command('ls')
    close()
