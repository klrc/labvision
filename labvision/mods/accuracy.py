from . import functional


class Accuracy():
    def __init__(self, testloader, model, topk=1, cuda=True, check_type_tensor=True):
        self.testloader = testloader
        self.model = model
        self.topk = topk
        self.cuda = cuda
        self.check_type_tensor = check_type_tensor
        self.history = []
        self._hooked_looper = None

    def __call__(self, info=None):
        acc = functional.accuracy(
            testloader=self.testloader,
            model=self.model,
            topk=self.topk,
            cuda=self.cuda,
            check_type_tensor=self.check_type_tensor,
        )
        if info is not None:
            acc = (info, acc)
        elif self._hooked_looper is not None:
            acc = (self._hooked_looper.axis(), acc)
        self.history.append(acc)
        return acc
