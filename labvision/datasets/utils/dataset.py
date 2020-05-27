import json
import os
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from .exceptions import DatasetNotFoundException


def __maxidx__(_list):
    return _list.index(max(_list))


class Dataset(data.Dataset):
    dataset_labels = ['single', 'distribution']
    img_dir = 'images'
    seed = 0

    def __init__(self, root, train=True, label_types='single', transform=None):
        """
            Basic dataset class,
            index should be like this:
                root_dir
                ->  img_dir
                ->  ground_truth.txt

            Args:
                root:
                train:
                label_types: str or tuple(str) for output types, multiple label types will be given as a tuple.
                transform:
        """

        self.init_label_types(label_types)
        self.transform = transform
        self.load_data(root, train)
        self.load_label(root)
        self.__totorch__()

    def __gensplit__(self, root):
        """
            generate train/test split as split.json,
            Args:
                root:
        """
        np.random.seed(self.seed)
        _list = [x for x in os.listdir(f'{root}/{self.img_dir}')]
        random.shuffle(_list)
        pivod = int(len(_list)*0.2)
        _dict = {'train': _list[pivod:], 'test': _list[:pivod]}
        with open(f'{root}/split.json', 'w') as f:
            json.dump(_dict, f)

    def __readgt__(self, root, spliter=' ', start_line=1, suffix=''):
        """
            read groundtruth from ground_truth.txt,
            (override if more complex func needed),
            Args:
                root:
                spliter: spliter for labels
                start_line: line to start from
                suffix: image file suffix
        """
        _dict = {}
        with open(f'{root}/ground_truth.txt', 'r') as f:
            for line in f.readlines()[start_line:]:  # images-name Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
                data = line.split(spliter)
                _dict[f'{data[0]}{suffix}'] = data
        return [_dict[x.split('/')[-1]] for x in self.data]

    def __totorch__(self):
        """
            turn targets into torch.Tensor,
        """
        for key in self.ys.keys():
            self.ys[key] = torch.from_numpy(np.array(self.ys[key]))

    def __cvimg__(self, path, convert2rgb=True):
        """
            read image in RGB mode.
        """
        img_cv = cv2.imread(path)
        if convert2rgb:
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
        if self.transform:
            img = self.transform(img)
        target = [self.ys[k][index] for k in self.ys.keys()]
        if len(target) == 1:
            target = target[0]
        return img, target

    def __len__(self):
        """
            function for torch.util.data.Dataset class.
        """
        return len(self.data)

    def init_label_types(self, label_types):
        if type(label_types) is str:
            label_types = [label_types]
        for x in label_types:
            assert x in self.dataset_labels
        self.ys = {x: [] for x in label_types}

    def load_data(self, root, train):
        """
            pre-load image path,
            (override if more complex func needed),
            Args:
                root:
                train:
        """
        if not os.path.exists(root):
            raise DatasetNotFoundException(root)
        _path = f'{root}/split.json'
        if not os.path.exists(_path):
            self.__gensplit__(root)
        with open(_path, 'r') as f:
            _json = json.load(f)
            self.data = [f'{root}/{self.img_dir}/{x}' for x in _json['train' if train else 'test']]

    def load_label(self, root):
        """
            sort groundtruth into different types,
            (override if more complex func needed),
            e.g.:
                'single': one-hot label (maximum),
                'distribution': distributional label (votes),
            Args:
                root:
        """
        for x in self.__readgt__(root):
            y = [int(_str.replace('\n', '')) for _str in x[1:]]
            if 'single' in self.ys.keys():
                self.ys['single'].append(__maxidx__(y))
            if 'distribution' in self.ys.keys():
                self.ys['distribution'].append(y)
