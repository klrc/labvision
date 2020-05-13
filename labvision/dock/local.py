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


def pack(source_dir, cache_dir, name='deploy_pack'):
    check_dir(cache_dir)
    target_fp = f'{cache_dir}/{name}.tar.gz'
    total = 0
    for _, _, files in os.walk(source_dir):  # 遍历统计
        for x in files:
            total += 1  # 统计文件夹下文件个数
    with tarfile.open(target_fp, "w:gz") as tar:
        current = 0
        for root, _, files in os.walk(source_dir):
            for fname in files:
                current += 1
                pathfile = f'{root}/{fname}'
                arcpath = f'{root.split(source_dir)[-1]}/{fname}'
                tar.add(pathfile, arcname=arcpath)
                print(f'compressing ({current}/{total}): {arcpath}')
    print(f'packing {target_fp}')
    return target_fp
