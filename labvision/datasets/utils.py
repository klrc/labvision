import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data
from labvision.transforms import no_transform
sys.path.append('.')


def __maxidx__(_list):
    return _list.index(max(_list))


class DatasetNotFoundException(Exception):
    def __init__(self, fp):
        super().__init__()
        self.fp = fp

    def __str__(self):
        return f'Dataset not found in {self.fp}, you can download from ?.'


class Dataset(data.Dataset):
    label_types = ['single', 'distribution']

    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.__loaddata__(root, train)
        self.ys = {x: [] for x in self.label_types}
        self.__loadtarget__(root)
        self.__totorch__()

    def __gensplit__(self, root, img_dir='images', seed=0):
        np.random.seed(seed)
        _list = [x for x in os.listdir(f'{root}/{img_dir}')]
        random.shuffle(_list)
        pivod = int(len(_list)*0.2)
        _dict = {'train': _list[pivod:], 'test': _list[:pivod]}
        with open(f'{root}/split.json', 'w') as f:
            json.dump(_dict, f)

    def __loaddata__(self, root, train, image_dir='images'):
        if not os.path.exists(root):
            raise DatasetNotFoundException(root)
        _path = f'{root}/split.json'
        if not os.path.exists(_path):
            self.__gensplit__(root)
        with open(_path, 'r') as f:
            _json = json.load(f)
            self.data = [f'{root}/{image_dir}/{x}' for x in _json['train' if train else 'test']]

    def __readgt__(self, root, spliter=' ', start_line=1, suffix=''):
        _dict = {}
        with open(f'{root}/ground_truth.txt', 'r') as f:
            for line in f.readlines()[start_line:]:  # images-name Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
                data = line.split(spliter)
                _dict[f'{data[0]}{suffix}'] = data
        return [_dict[x.split('/')[-1]] for x in self.data]

    def __loadtarget__(self, root, spliter=' ', start_line=1):
        for x in self.__readgt__(root):
            y = [int(_str.replace('\n', '')) for _str in x[1:]]
            self.ys['single'].append(__maxidx__(y))
            self.ys['distribution'].append(y)

    def __totorch__(self):
        for key in self.ys.keys():
            self.ys[key] = torch.from_numpy(np.array(self.ys[key]))

    def __cvimg__(self, path):
        img_cv = cv2.imread(path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv

    def __getitem__(self, index):
        img = self.__cvimg__(self.data[index])
        img = no_transform()(img) if self.transform is None else self.transform(img)
        target = [self.ys[k][index] for k in self.ys.keys()]
        return img, target

    def __len__(self):
        return len(self.data)
