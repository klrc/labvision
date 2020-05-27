import torch

from . import functional


class TrainLooper():
    def __init__(self, trainloader, model, criterion=None, optimizer=None,
                 max_epoch=None, start_epoch=0):
        self.trainloader = trainloader
        self.model = model
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion
        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self._looper = None

    def _main_loop(self):
        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            for iteration, batch in enumerate(self.trainloader):  # batch loop
                loss = functional.train_step(  # one step forward & backward
                    batch=batch,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    cuda=self.cuda,
                    check_type_tensor=self.check_type_tensor
                )
                self.iteration = iteration
                yield loss
            epoch += 1
            if epoch > self.max_epoch:
                return

    def step(self, iteration=0, epoch=0):
        if self._looper is None:
            self._looper = self._main_loop()
        if epoch < 1:
            iteration = int(len(self.trainloader)*epoch)  # convert float epoch to iterations
            epoch = 0
        if iteration + epoch == 0:
            iteration = 1  # step for at least 1 iteration.

        target_epoch = self.epoch + epoch
        while target_epoch > self.epoch:
            loss = next(self._looper)
        full_iters = len(self.trainloader)
        for _ in range(iteration):
            loss = next(self._looper)
            if self.iteration + 1 >= full_iters:
                break
        return loss

    def __iter__(self):
        yield self.step(epoch=1)


class ValLooper():
    def __init__(self, valloader, model, criterion=None):
        self.valloader = valloader
        self.model = model
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion
        self._looper = None

    def _main_loop(self):
        while True:
            for batch in self.valloader:  # batch loop
                loss = functional.validation_step(  # one step forward
                    batch=batch,
                    model=self.model,
                    criterion=self.criterion,
                    cuda=self.cuda,
                    check_type_tensor=self.check_type_tensor
                )
                yield loss

    def step(self, iteration=4, epoch=0):
        if self._looper is None:
            self._looper = self._main_loop()

        if epoch > 0:
            iteration = int(len(self.trainloader)*epoch)  # convert epoch to iterations

        loss = 0.0
        for _ in range(iteration):
            loss += next(self._looper)
        return loss/iteration
