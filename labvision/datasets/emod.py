import scipy.io as scio
import torch
import numpy as np

from .utils import Dataset


def __maxidx__(_list):
    return _list.index(max(_list))


class EMOd(Dataset):
    dataset_labels = ['single', 'distribution', 'vad',  'eyetrack']
    img_dir = 'EMOdImages1019'

    def __init__(self, root, mask_transform, **kwargs):
        super().__init__(root=root, **kwargs)
        self.mask_transform = mask_transform

    def __readgt__(self, root):
        _dict = {}
        # for train/test split and some inaccessable data from IAPS
        for x in scio.loadmat(f'{root}/allfindata1019_renamed.mat')['allfindata1019']:
            _dict[int(x[0])] = x
        return [_dict[int(x.split('/')[-1].replace('.jpg', ''))] for x in self.data]

    def __totorch__(self):
        for key in self.ys.keys():
            if key != 'eyetrack':
                self.ys[key] = torch.from_numpy(np.array(self.ys[key]))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if 'eyetrack' in self.ys.keys():
            if len(self.ys.keys()) > 1:
                i = [x for x in self.ys.keys()].index('eyetrack')
                _eyetrack = target[i]
                _eyetrack = self.__cvimg__(_eyetrack)
                _eyetrack = self.mask_transform(_eyetrack)
                _eyetrack = torch.mean(_eyetrack, dim=0)
                target[i] = _eyetrack
            else:
                _eyetrack = target
                _eyetrack = self.__cvimg__(_eyetrack)
                _eyetrack = self.mask_transform(_eyetrack)
                _eyetrack = torch.mean(_eyetrack, dim=0)
                target = _eyetrack
        return img, target

    def load_label(self, root):
        for index, x in enumerate(self.__readgt__(root)):
            y1 = [float(_str) for _str in x[3:6]]
            y2 = [float(_str) for _str in x[6:15]]
            if 'single' in self.ys.keys():
                self.ys['single'].append(__maxidx__(y2))
            if 'distribution' in self.ys.keys():
                self.ys['distribution'].append(y2)
            if 'vad' in self.ys.keys():
                self.ys['vad'].append(y1)
            if 'eyetrack' in self.ys.keys():
                self.ys['eyetrack'].append(self.data[index].replace('EMOdImages1019', 'FixationMap/Continous_map'))
