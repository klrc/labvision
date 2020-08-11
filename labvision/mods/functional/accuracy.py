import torch
from .step import _check_batch_data


def _correct_topk(inputs: torch.Tensor, target: torch.Tensor, k: int):
    if k > 1:
        inputs = inputs.topk(k, 1, True, True)[1]
        target = target.view(-1, 1)
    else:
        _, inputs = torch.max(inputs, 1)
    return inputs.eq(target).sum().float().item()


def accuracy(testloader: torch.utils.data.DataLoader, model: torch.nn.Module, topk=1, cuda=True, check_type_tensor=True, distribution_input=False):
    with torch.no_grad():
        if model.training:
            model.eval()
        correct = 0.0
        total = 0.0
        for batch in testloader:
            x, target = _check_batch_data(batch, cuda=cuda, check_type_tensor=check_type_tensor)
            if distribution_input:
                target = target.argmax(dim=1)
            outputs = model(x)
            total += target.size(0)
            correct += _correct_topk(outputs, target, topk)
    acc = correct / total
    return acc
