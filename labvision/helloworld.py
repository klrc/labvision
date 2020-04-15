import torch


def hellotorch():
    print(f'hello torch, cuda={torch.cuda.is_available()}')
