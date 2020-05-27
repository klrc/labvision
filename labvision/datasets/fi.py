import os
import numpy as np
import random
from .utils import Dataset
import json


def __maxidx__(_list):
    return _list.index(max(_list))


class FI(Dataset):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)

    def __gensplit__(self, root):
        np.random.seed(self.seed)
        _list = [x for x in os.listdir(f'{root}/{self.img_dir}')]
        random.shuffle(_list)
        data = {}
        for x in _list:
            label = x.split('_')[0]
            if label not in data.keys():
                data[label] = []
            data[label].append(x)
        _dict = {'train': [], 'test': []}
        for key in data.keys():
            _list = data[key]
            pivod = int(len(_list)*0.15)
            _dict['train'].extend(_list[pivod:])
            _dict['test'].extend(_list[:pivod])
        random.shuffle(_dict['train'])
        random.shuffle(_dict['test'])
        with open(f'{root}/split.json', 'w') as f:
            json.dump(_dict, f)

    def load_label(self, root, spliter=' ', start_line=1):
        labels = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness']
        for x in self.data:
            label = x.split('/')[-1].split('_')[0]
            y = [1 if _str == label else 0 for _str in labels]
            if 'single' in self.ys.keys():
                self.ys['single'].append(__maxidx__(y))
            if 'distribution' in self.ys.keys():
                self.ys['distribution'].append(y)
