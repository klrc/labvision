import torch


class AutoControl():
    def __init__(self, dataset, model=None,
                 dataset_transforms=None, dataset_args={}, optimizer=torch.optim.SGD, optimizer_args={},
                 forward_func=None, loss_func=None, **fpargs):
        self._init_datasets(dataset, dataset_transforms, **dataset_args)

    def _init_datasets(self, dataset, dataset_transforms, batch_size=32, num_workers=2, **args):
        """
            Args:
                dataset:
                dataset_transforms: transform (for all) or Tuple(transform_train, transform_test)
        """
        if type(dataset_transforms) is tuple:
            _transform_train, _transform_test = dataset_transforms
        else:
            _transform_train = _transform_test = dataset_transforms
        trainset = dataset(train=True, transform=_transform_train, **args)
        testset = dataset(train=False, transform=_transform_test, **args)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.evalloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    import labvision
    ac = AutoControl(labvision.datasets.Dataset)
    # print(type(None) is tuple)
