from .utils import Dataset


def __maxidx__(_list):
    return _list.index(max(_list))


class EmotionROI(Dataset):
    dataset_labels = ['single', 'distribution', 'va']

    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)

    def __readgt__(self, root):
        return super().__readgt__(root, spliter='\t')

    def load_label(self, root):
        for x in self.__readgt__(root):
            y1 = [float(_str.replace('\n', '')) for _str in x[3:]]
            y2 = [float(_str.replace('\n', '')) for _str in x[1:3]]
            if 'single' in self.ys.keys():
                self.ys['single'].append(__maxidx__(y1))
            if 'distribution' in self.ys.keys():
                self.ys['distribution'].append(y1)
            if 'va' in self.ys.keys():
                self.ys['va'].append(y2)
