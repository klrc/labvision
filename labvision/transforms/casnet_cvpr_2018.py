from ._basic import basic_resize


def casnet_fixation_transform():
    return basic_resize(size=(18, 25), with_normalize=False)
