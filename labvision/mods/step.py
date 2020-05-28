from . import functional


class Train():
    def __init__(self, model, criterion, optimizer, cuda=True, check_type_tensor=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cuda = cuda
        self.check_type_tensor = check_type_tensor

    def __call__(self, batch, info=None):
        loss = functional.train_step(
            batch=batch,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            cuda=self.cuda,
            check_type_tensor=self.check_type_tensor,
        )
        return loss


class Validation():
    def __init__(self, model, criterion, cuda=True, check_type_tensor=False):
        self.model = model
        self.criterion = criterion
        self.cuda = cuda
        self.check_type_tensor = check_type_tensor

    def __call__(self, batch, info=None):
        loss = functional.validation_step(
            batch=batch,
            model=self.model,
            criterion=self.criterion,
            cuda=self.cuda,
            check_type_tensor=self.check_type_tensor,
        )
        return loss
