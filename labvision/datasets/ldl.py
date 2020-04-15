from .utils import Dataset


class FlickrLDL(Dataset):
    def __init__(self, root='external/Flickr_LDL', train=True, transform=None):
        super().__init__(root, train, transform)


class TwitterLDL(Dataset):
    def __init__(self, root='external/Twitter_LDL', train=True, transform=None):
        super().__init__(root, train, transform)
