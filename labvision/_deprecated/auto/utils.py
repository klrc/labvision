import torch
import numpy as np
import random


def manual_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    # https://zhuanlan.zhihu.com/p/76472385
    # torch.backends.cudnn.benchmark = False
