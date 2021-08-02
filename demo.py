# Based on: kuangliu/pytorch-cifar

import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

from utils import AverageMeter
from tqdm import trange


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion*planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


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

# Plot training
fig, axs = plt.subplots(2, 2, figsize=(25, 20))

axs[0, 0].set_title('Train Loss')
axs[0, 1].set_title('Training Accuracy')
axs[1, 0].set_title('Test Loss')
axs[1, 1].set_title('Test Accuracy')

axs[0, 0].plot(train_loss)
axs[0, 1].plot(train_accuracy)
axs[1, 0].plot(test_loss)
axs[1, 1].plot(test_accuracy)

fig.savefig("plots/results.png")
