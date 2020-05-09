

from .utils import ChainedTransform as C_


to_tensor = C_().ToTensor()
empty_transform = C_().ToPILImage().ToTensor()


def resize_rdcrop_flip(size1, size2, **args):
    """
        Args:
            size1: resize target size.
            size2: rdcrop target size.
    """
    x = C_().ToPILImage()
    x = x.Resize(size=size1, **args).RandomCrop(size=size2, **args).RandomHorizontalFlip().ToTensor()
    return x


def resize_centercrop_flip(size1, size2,  **args):
    """
        Args:
            size1: resize target size.
            size2: centercrop target size.
    """
    x = C_().ToPILImage()
    x = x.Resize(size=size1, **args).CenterCrop(size=size2, **args).RandomHorizontalFlip().ToTensor()
    return x
