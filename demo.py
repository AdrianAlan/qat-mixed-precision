# Based on: kuangliu/pytorch-cifar

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

from models import ResNet18
from utils import AverageMeter, Plotting
from tqdm import trange


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Prepare the ResNet18 model
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Training
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def run_epoch(epoch,
              net,
              loader,
              prefix='Training',
              backprop=True,
              fp16=False):
    tr = trange(len(loader), file=sys.stdout)
    loss = AverageMeter('Loss', ':1.5f')
    accuracy = AverageMeter('Accuracy', ':1.5f')
    loss.reset()
    accuracy.reset()
    for (inputs, targets) in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if fp16:
            inputs = inputs.half()
        optimizer.zero_grad()
        outputs = net(inputs)
        batch_loss = criterion(outputs, targets)
        if backprop:
            batch_loss.backward()
            optimizer.step()
        loss.update(batch_loss.item(), targets.size(0))
        _, predicted = outputs.max(1)
        accuracy.update(predicted.eq(targets).sum().item()/targets.size(0),
                        targets.size(0))
        tr.set_description('{0} epoch {1}: {2}, {3}'.format(
            prefix, epoch, loss, accuracy))
        tr.update(1)
    tr.close()
    return loss.average, accuracy.average,


def run(epoch, train=True, fp16=False):
    if train:
        if fp16:
            net.float()
        net.train()
        loss, accuracy = run_epoch(epoch, net, train_loader)
    else:
        net.eval()
        if fp16:
            net.half()
        with torch.no_grad():
            loss, accuracy = run_epoch(epoch,
                                       net,
                                       test_loader,
                                       prefix='Test',
                                       backprop=False,
                                       fp16=fp16)
    return loss, accuracy


train_accuracy, test_accuracy = [], []
train_loss, test_loss = [], []

fp16 = True

best_accuracy = 0

for epoch in range(1, 20):

    loss, accuracy = run(epoch, train=True, fp16=fp16)
    train_loss.append(loss)
    train_accuracy.append(accuracy)

    loss, accuracy = run(epoch, train=False, fp16=fp16)
    test_loss.append(loss)
    test_accuracy.append(accuracy)

    if accuracy > best_accuracy:
        state = {
            'accuracy': accuracy,
            'epoch': epoch,
            'net': net.state_dict()
        }
        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(net.state_dict(), './models/checkpoint.pth')
        best_accuracy = accuracy
    scheduler.step()

plot = Plotting('plots/results.png')
plot.draw(train_loss, train_accuracy, test_loss, test_accuracy)
