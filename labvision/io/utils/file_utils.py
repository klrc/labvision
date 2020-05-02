import shutil
import tarfile
import os

import re


def collect_hash(lines):
    _cache = []
    for line in lines:
        res = re.search(r'^(?P<value>\w(\w|\d){7})(\.|\-)', line)
        if res is None:
            continue
        res = res.groupdict()['value']
        if res in _cache:
            continue
        _cache.append(res)
        yield res


def clean(source_dir='build', target_dir='dist'):
    check_dir(target_dir)
    source_files = [x for x in os.listdir(source_dir)]
    for x in collect_hash(source_files):
        tar_path = f'{target_dir}/{x}.tar.gz'
        print(f'compress: {tar_path}')
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, _, files in os.walk(source_dir):
                for fname in files:
                    if x in fname:
                        pathfile = os.path.join(root, fname)
                        tar.add(pathfile)
        for fname in source_files:
            if x in fname:
                rm_path = f'{source_dir}/{fname}'
                print(f'remove: {rm_path}')
                if os.path.isdir(rm_path):
                    shutil.rmtree(rm_path, True)
                else:
                    os.remove(rm_path)
    print('clean complete.')


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'created dir: {path}')


if __name__ == "__main__":
    pass
    # clean('/home/sh/Desktop/Research/external/backup', '/home/sh/Desktop/Research/dist')
    # pack('/home/sh/Desktop/Research/src', '/home/sh/Desktop/Research/build/deploy.tar.gz')
    # deploy('/home/sh/Desktop/Research/build/deploy.tar.gz')
