from .utils import Dataset


class IAPS(Dataset):
    dataset_labels = ['vad']

    def __init__(self, root='external/IAPS/IAPS', **kwargs):
        if 'label_types' in kwargs.keys():
            print('only vad labels are available in IAPS/NAPS, please check your args.')
            raise NotImplementedError
        super().__init__(root=root, label_types='vad', **kwargs)

    def __readgt__(self, root):
        return super().__readgt__(root, spliter='\t', start_line=0, suffix='.jpg')

    def load_label(self, root):
        for x in self.__readgt__(root):
            y = [float(_str.replace('\n', '')) for _str in x[1:]]
            self.ys['vad'].append(y)


class NAPS(IAPS):
    def __init__(self, root='external/IAPS/NAPS', **kwargs):
        super().__init__(root, **kwargs)
