from .utils import Dataset


class IAPS(Dataset):
    label_types = ['vad']

    def __init__(self, root='external/IAPS/IAPS', train=True, transform=None):
        super().__init__(root, train, transform)

    def __loadtarget__(self, root):
        self.ys = {'vad': []}
        for x in self.__readgt__(root, spliter='\t', start_line=0, suffix='.jpg'):
            y = [float(_str.replace('\n', '')) for _str in x[1:]]
            self.ys['vad'].append(y)


class NAPS(IAPS):
    def __init__(self, root='external/IAPS/NAPS', train=True, transform=None):
        super().__init__(root, train, transform)
