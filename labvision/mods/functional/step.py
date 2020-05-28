import torch


def _check_batch_data(batch, cuda, check_type_tensor):
    x, target = batch
    if check_type_tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
    if cuda:
        x = x.cuda()
        target = target.cuda()
    return x, target


def train_step(batch, model: torch.nn.Module, criterion, optimizer: torch.optim.Optimizer, cuda=True, check_type_tensor=True):
    x, target = _check_batch_data(batch, cuda=cuda, check_type_tensor=check_type_tensor)  # check inputs
    if not model.training:  # switch model mode
        model.train()
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def validation_step(batch, model: torch.nn.Module, criterion=None, cuda=True, check_type_tensor=True):
    x, target = _check_batch_data(batch, cuda=cuda, check_type_tensor=check_type_tensor)  # check inputs
    _ret = None
    with torch.no_grad():  # forward without gradient
        if model.training:  # switch model mode
            model.eval()
        y = model(x)
        if criterion:  # return the loss if a criterion function is specified.
            loss = criterion(y, target)
            _ret = loss.item()
        else:
            _ret = y.data
    return _ret
