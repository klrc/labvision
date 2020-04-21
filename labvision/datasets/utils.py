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
        raise NotImplementedError  # TODO: add download src
        return f'Dataset not found in {self.fp}, you can download from ?.'


class Dataset(data.Dataset):
    label_types = ['single', 'distribution']

    def __init__(self, root, train=True, transform=None):
        """
            Basic dataset class,
            index should be like this:
                root_dir
                ->  img_dir
                ->  ground_truth.txt

            Args:
                root:
                train:
                transform:
        """
        self.transform = transform
        self.__loaddata__(root, train)
        self.ys = {x: [] for x in self.label_types}
        self.__loadtarget__(root)
        self.__totorch__()

    def __gensplit__(self, root, img_dir='images', seed=0):
        """
            generate train/test split as split.json,
            Args:
                root:
                img_dir:
                seed:
        """
        np.random.seed(seed)
        _list = [x for x in os.listdir(f'{root}/{img_dir}')]
        random.shuffle(_list)
        pivod = int(len(_list)*0.2)
        _dict = {'train': _list[pivod:], 'test': _list[:pivod]}
        with open(f'{root}/split.json', 'w') as f:
            json.dump(_dict, f)

    def __loaddata__(self, root, train, img_dir='images'):
        """
            pre-load image path,
            Args:
                root:
                train:
                img_dir:

            override if more complex func needed.
        """
        if not os.path.exists(root):
            raise DatasetNotFoundException(root)
        _path = f'{root}/split.json'
        if not os.path.exists(_path):
            self.__gensplit__(root)
        with open(_path, 'r') as f:
            _json = json.load(f)
            self.data = [f'{root}/{img_dir}/{x}' for x in _json['train' if train else 'test']]

    def __readgt__(self, root, spliter=' ', start_line=1, suffix=''):
        """
            read groundtruth from ground_truth.txt,
            Args:
                root:
                spliter: spliter for labels
                start_line: line to start from
                suffix: image file suffix

            override if more complex func needed.
        """
        _dict = {}
        with open(f'{root}/ground_truth.txt', 'r') as f:
            for line in f.readlines()[start_line:]:  # images-name Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
                data = line.split(spliter)
                _dict[f'{data[0]}{suffix}'] = data
        return [_dict[x.split('/')[-1]] for x in self.data]

    def __loadtarget__(self, **params):
        """
            sort groundtruth into different types,
            'single': one-hot label (maximum),
            'distribution': distributional label (votes),

            Args:
                see __readgt__()

            override if more complex func needed.
        """
        for x in self.__readgt__(**params):
            y = [int(_str.replace('\n', '')) for _str in x[1:]]
            self.ys['single'].append(__maxidx__(y))
            self.ys['distribution'].append(y)

    def __totorch__(self):
        """
            turn targets into torch.Tensor,
        """
        for key in self.ys.keys():
            self.ys[key] = torch.from_numpy(np.array(self.ys[key]))

    def __cvimg__(self, path):
        """
            read image in RGB mode.
        """
        img_cv = cv2.imread(path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv

    def __getitem__(self, index):
        """
            function for torch.util.data.Dataset class,
            returns image, (target_1, target_2, ..., target_n)

            Args:
                index:
        """
        img = self.__cvimg__(self.data[index])
        img = no_transform()(img) if self.transform is None else self.transform(img)
        target = [self.ys[k][index] for k in self.ys.keys()]
        return img, target

    def __len__(self):
        """
            function for torch.util.data.Dataset class.
        """
        return len(self.data)
