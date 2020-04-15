from .utils import Dataset


def __maxidx__(_list):
    return _list.index(max(_list))


class EmotionROI(Dataset):
    label_types = ['single', 'distribution', 'va']

    def __init__(self, root='external/EmotionROI', train=True, transform=None):
        super().__init__(root, train, transform)

    def __loadtarget__(self, root):
        for x in self.__readgt__(root, spliter='\t'):
            y1 = [float(_str.replace('\n', '')) for _str in x[3:]]
            y2 = [float(_str.replace('\n', '')) for _str in x[1:3]]
            self.ys['single'].append(__maxidx__(y1))
            self.ys['distribution'].append(y1)
            self.ys['va'].append(y2)
