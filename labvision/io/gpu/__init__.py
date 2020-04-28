

import torch


device_id = None
memory_limit = None


def cuda_available():
    return torch.cuda.is_available()


def limit():
    print('gpu limit func.')
    raise NotImplementedError
