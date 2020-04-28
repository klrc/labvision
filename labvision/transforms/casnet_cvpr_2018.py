from .utils import ChainedTransform


def casnet_fixation_transform():
    x = ChainedTransform()
    x = x.ToPILImage().Resize(size=(18, 25))
    x = x.ToTensor()
    return x
