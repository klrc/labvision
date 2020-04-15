

from .chained_transform import ChainedTransform

__all__ = [
    'no_transform', 'basic_normalize', 'basic_resize', 'basic_resize_rdcrop',
]


def no_transform():
    x = ChainedTransform()
    x = x.ToPILImage().ToTensor()
    return x.Compose()


def basic_normalize(**args):
    x = ChainedTransform()
    x = x.ToPILImage().ToTensor().Normalize(**args)
    return x.Compose()


def basic_resize(with_normalize=True, **args):
    x = ChainedTransform()
    x = x.ToPILImage().Resize(**args).ToTensor()
    if with_normalize:
        x = x.Normalize()
    return x.Compose()


def basic_resize_rdcrop(size1, size2, with_normalize=True, **args):
    x = ChainedTransform()
    x = x.ToPILImage().Resize(size=size1, **args).RandomCrop(size=size2, **args).ToTensor()
    if with_normalize:
        x = x.Normalize()
    return x.Compose()
