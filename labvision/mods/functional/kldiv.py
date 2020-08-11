import torch
from .step import _check_batch_data


def kldiv(testloader: torch.utils.data.DataLoader, model: torch.nn.Module, cuda=True, check_type_tensor=True):
    with torch.no_grad():
        if model.training:
            model.eval()
        kl = 0.0
        total = 0.0
        for batch in testloader:
            x, target = _check_batch_data(batch, cuda=cuda, check_type_tensor=check_type_tensor)
            outputs = model(x)
            total += target.size(0)
            outputs = torch.nn.functional.log_softmax(outputs)
            target = torch.nn.functional.softmax(target)
            kl += torch.nn.functional.kl_div(outputs, target)
    kl = kl / total
    return kl
