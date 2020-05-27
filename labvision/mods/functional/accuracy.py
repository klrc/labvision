import torch
from .step import _check_batch_data


def _correct_topk(inputs, target, k):
    return (inputs.topk(max((1, k)), 1, True, True)[1] == target.view(-1, 1)).sum().float().item()


def accuracy(testloader, model, topk=1, cuda=True, check_type_tensor=True):
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in testloader:
            x, target = _check_batch_data(batch, cuda=cuda, check_type_tensor=check_type_tensor)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            if topk > 1:
                correct += _correct_topk(predicted, target, topk)
            else:
                correct += predicted.eq(target).sum().float().item()
    acc = correct / total
    return acc
