# Based on: kuangliu/pytorch-cifar

import torch
import torchvision
import torchvision.transforms as transforms


# Prepare CIFAR10 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train),
    batch_size=128,
    shuffle=True,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test),
    batch_size=100,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
