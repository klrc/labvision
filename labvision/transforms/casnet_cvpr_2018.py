from .chained_transform import ChainedTransform


def casnet_fixation_transform():
    x = ChainedTransform()
    x = x.ToPILImage().Resize((18, 25))
    x = x.ToTensor()
    return x
