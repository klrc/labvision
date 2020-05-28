from torch.utils.data import DataLoader
from .dry import Dry


def DataLoaders(root, dataset_class, transform_train, transform_test=None, batch_size=32, num_workers=2, pin_memory=False, compile=True, logger=None):
    if transform_test is None:
        transform_test = transform_train

    trainset = Dry.Dataset(dataset_class)  # create dry dataset parameters.
    trainset.root = root
    trainset.train = True
    trainset.transform = transform_train
    testset = Dry.Dataset(dataset_class)  # create dry dataset parameters.
    testset.root = root
    testset.train = False
    testset.transform = transform_test
    valset = Dry.Dataset(dataset_class)
    valset.root = root
    valset.train = False
    valset.transform = transform_test

    trainloader = Dry.Args(DataLoader)  # init trainloader
    trainloader.dataset = trainset
    trainloader.batch_size = batch_size
    trainloader.shuffle = True
    trainloader.num_workers = num_workers
    trainloader.pin_memory = pin_memory

    valloader = Dry.Args(DataLoader)  # init valloader
    valloader.dataset = valset
    valloader.batch_size = batch_size
    valloader.shuffle = True
    valloader.num_workers = num_workers
    valloader.pin_memory = pin_memory

    testloader = Dry.Args(DataLoader)  # init testloader
    testloader.dataset = testset
    testloader.batch_size = batch_size
    testloader.shuffle = False
    testloader.num_workers = num_workers
    testloader.pin_memory = pin_memory

    if compile:
        trainloader = trainloader.compile(logger=logger)
        valloader = valloader.compile(logger=logger)
        testloader = testloader.compile(logger=logger)

    return trainloader, valloader, testloader
