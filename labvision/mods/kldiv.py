from . import functional


class KLDiv():
    def __init__(self, testloader, model, cuda=True, check_type_tensor=True):
        self.testloader = testloader
        self.model = model
        self.cuda = cuda
        self.check_type_tensor = check_type_tensor
        self.history = []
        self._hooked_looper = None

    def __call__(self, info=None):
        kl = functional.kldiv(
            testloader=self.testloader,
            model=self.model,
            cuda=self.cuda,
            check_type_tensor=self.check_type_tensor,
        )
        if info is not None:
            kl = (info, kl)
        elif self._hooked_looper is not None:
            kl = (self._hooked_looper.axis(), kl)
        self.history.append(kl)
        return kl
