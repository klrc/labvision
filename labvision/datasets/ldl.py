from .utils import Dataset


class FlickrLDL(Dataset):
    def __init__(self, root='external/Flickr_LDL', **kwargs):
        super().__init__(root, **kwargs)


class TwitterLDL(Dataset):
    def __init__(self, root='external/Twitter_LDL', **kwargs):
        super().__init__(root, **kwargs)
