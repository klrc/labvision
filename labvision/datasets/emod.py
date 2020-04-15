import scipy.io as scio
import torch
import numpy as np

from .utils import Dataset
from labvision.transforms import casnet_fixation_transform


def __maxidx__(_list):
    return _list.index(max(_list))


class EMOd(Dataset):
    label_types = ['single', 'distribution', 'vad',  'eyetrack']

    def __init__(self, root='external/EMOd', train=True, transform=None, mask_transform=None, eyetrack_map=True):
        self.eyetrack_map = eyetrack_map
        self.mask_transform = casnet_fixation_transform() if mask_transform is None else mask_transform
        super().__init__(root, train, transform)

    def __gensplit__(self, root):
        super().__gensplit__(root, 'EMOdImages1019')

    def __readgt__(self, root):
        _dict = {}
        # for train/test split and some inaccessable data from IAPS
        for x in scio.loadmat(f'{root}/allfindata1019_renamed.mat')['allfindata1019']:
            _dict[int(x[0])] = x
        return [_dict[int(x.split('/')[-1].replace('.jpg', ''))] for x in self.data]

    def __loaddata__(self, root, train):
        super().__loaddata__(root, train, image_dir='EMOdImages1019')

    def __loadtarget__(self, root, spliter=' ', start_line=1):
        for index, x in enumerate(self.__readgt__(root)):
            y1 = [float(_str) for _str in x[3:6]]
            y2 = [float(_str) for _str in x[6:15]]
            self.ys['single'].append(__maxidx__(y2))
            self.ys['distribution'].append(y2)
            self.ys['vad'].append(y1)
            self.ys['eyetrack'].append(self.data[index].replace('EMOdImages1019', 'FixationMap/Continous_map'))

    def __totorch__(self):
        for key in self.ys.keys():
            if key != 'eyetrack':
                self.ys[key] = torch.from_numpy(np.array(self.ys[key]))

    def __getitem__(self, index):
        img = self.__cvimg__(self.data[index])
        img, target = super().__getitem__(index)
        _single, _distribution, _vad, _eyetrack = target
        if self.eyetrack_map:
            _eyetrack = self.__cvimg__(_eyetrack)
            _eyetrack = self.mask_transform(_eyetrack)
            _eyetrack = torch.mean(_eyetrack, dim=0)
        return img, [_single, _distribution, _vad, _eyetrack]
