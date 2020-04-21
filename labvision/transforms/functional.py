

from .chained_transform import ChainedTransform

__all__ = [
    'empty_transform', 'resize_rdcrop_flip', 'resize_centercrop_flip',
]


def empty_transform():
    x = ChainedTransform()
    x = x.ToPILImage().ToTensor()
    return x


def resize_rdcrop_flip(size1, size2, **args):
    """
        Args:
            size1: resize target size.
            size2: rdcrop target size.
    """
    x = ChainedTransform().ToPILImage()
    x = x.Resize(size=size1, **args).RandomCrop(size=size2, **args).RandomHorizontalFlip().ToTensor()
    return x


def resize_centercrop_flip(size1, size2,  **args):
    """
        Args:
            size1: resize target size.
            size2: centercrop target size.
    """
    x = ChainedTransform().ToPILImage()
    x = x.Resize(size=size1, **args).CenterCrop(size=size2, **args).RandomHorizontalFlip().ToTensor()
    return x
