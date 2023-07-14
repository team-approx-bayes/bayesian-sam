import os
import torch
import torchvision
from torchvision import transforms

def dataloader(datasetname : str):
    """ returns dataloaders for a given dataset """

    datasets = { 'mnist': _mnist, 
                 'fashion': _fashion, 
                 'cifar10': _cifar10, 
                 'cifar100': _cifar100,
                 'tinyimagenet': _tinyimagenet }

    return datasets[datasetname.lower()]

def _mnist(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = False, nworkers : int = 8):
    """ MNIST, 60000 28x28x1 images, 10 classes, 10000 test images """

    transform_totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=datasetfolder, train=True, 
        download=True, transform=transform_totensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers)

    testset = torchvision.datasets.MNIST(
        root=datasetfolder, train=False, 
        download=True, transform=transform_totensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers)

    return trainset, testset, trainloader, testloader

def _fashion(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = False, nworkers : int = 8):
    """ FashionMNIST, 60000 28x28x1 images, 10 classes, 10000 test images """

    transform_totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root=datasetfolder, train=True, 
        download=True, transform=transform_totensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers)

    testset = torchvision.datasets.FashionMNIST(
        root=datasetfolder, train=False, 
        download=True, transform=transform_totensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers)

    return trainset, testset, trainloader, testloader

def _cifar10(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = True, nworkers : int = 8):
    """ CIFAR-10, 50000 32x32x3 images, 10 classes, 10000 test images """

    train_mean = (0.4914, 0.4822, 0.4465)
    train_std = (0.2023, 0.1994, 0.2010)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=datasetfolder, train=True, 
        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root=datasetfolder, train=False, 
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader

def _cifar100(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = True, nworkers : int = 8):
    """ CIFAR100, 50000 32x32x3 images, 100 classes, 10000 test images """

    train_mean = (0.5071, 0.4865, 0.4409)
    train_std = (0.2673, 0.2564, 0.2761)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=datasetfolder, train=True, 
        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.CIFAR100(
        root=datasetfolder, train=False, 
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader

def _tinyimagenet(batchsize : int, testbatchsize : int, datasetfolder : str, 
                  augment : bool = True, nworkers : int = 8):
    """ TinyImageNet, 100000 64x64x3 images, 200 classes, 10000 test images """

    train_mean = (0.485, 0.456, 0.406)
    train_std = (0.229, 0.224, 0.225)

    data_dir = os.path.join(datasetfolder, 'tiny-imagenet-200')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'val/images')

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, 
        shuffle=True, num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.ImageFolder(
        valid_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, 
        shuffle=False, num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader
