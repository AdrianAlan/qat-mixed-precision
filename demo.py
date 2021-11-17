# Based on: kuangliu/pytorch-cifar

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append(r"/home/olyas/ailabs")
from ailabs_qat.model.retrainer import QatRetrainerModel
from models import ResNet18
from utils import AverageMeter, Plotting
from tqdm import trange


def get_cifar10_loaders(train_batch_size=256, test_batch_size=512):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                ])),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                ])),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader, test_loader


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
        if backprop:
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
    print("accuracy: {}".format(accuracy.average))
    return loss.average, accuracy.average


def run(epoch, loader,  train=True, fp16=False):
    if train:
        if fp16:
            net.float()
        net.train()
        loss, accuracy = run_epoch(epoch, net, loader)
    else:
        net.eval()
        if fp16:
            net.half()
        with torch.no_grad():
            loss, accuracy = run_epoch(epoch,
                                       net,
                                       loader,
                                       prefix='Test',
                                       backprop=False,
                                       fp16=fp16)
    return loss, accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train ResNet18 on CIFAR10')
    parser.add_argument('name',
                        type=str,
                        help='Model name')
    parser.add_argument('-b',
                        type=int,
                        help='Training batch size',
                        default=256,
                        dest='batch_size')
    parser.add_argument('-e',
                        type=int,
                        help='Training epochs',
                        default=20,
                        dest='epochs')
    parser.add_argument('--lr',
                        type=float,
                        help='Learning rate',
                        default=0.01,
                        dest='learning_rate')
    parser.add_argument('--momentum',
                        type=float,
                        help='SGD momentum',
                        default=0.9,
                        dest='momentum')
    parser.add_argument('--decay',
                        type=float,
                        help='SGD weight decay',
                        default=5e-4,
                        dest='weight_decay')
    parser.add_argument('--fp16',
                        action="store_true",
                        help='Inference in half precision')
    parser.add_argument('--qat',
                        action="store_true",
                        help='Quantization aware training')
    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        help='Output verbosity')
    parser.add_argument('--qonnx',
                        action="store_true",
                        help='Export Qat to onnx')

    args = parser.parse_args()

    # Creating directories
    for d in ['models', 'plots']:
        if not os.path.isdir(d):
            os.mkdir(d)

    # Prepare CIFAR10 dataset
    train_loader, test_loader = get_cifar10_loaders(
        train_batch_size=args.batch_size)

    # Prepare the ResNet18 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load("models/float_best.pth"))
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    #print("Float accuracy: ")
    #loss, accuracy = run(0, test_loader, train=False, fp16=args.fp16)
    net = net.module
    if args.qat:
        net = QatRetrainerModel(net, qat_config='qat_config.yaml')
        net.print_configurable_layers_id()
        net.set_lock_activation_quantization(False)
        net.to(device)
    if args.qonnx:
        dummy = torch.rand((1,3,30,30)).cuda()
        net(dummy)
        net.export_to_onnx(dummy, 'models/4bit_model.onnx')
    #net.print_configurable_layers_id() # print layers names for configuration
    print("8 bit accuracy: ")
    loss, accuracy = run(0, test_loader, train=False, fp16=args.fp16)

    if device == 'cuda':
        cudnn.benchmark = True
        net = torch.nn.DataParallel(net)

    # Training

    params_sc = [p for n, p in net.named_parameters() if n.endswith("_sc")]
    params_other = [p for n, p in net.named_parameters() if not n.endswith("_sc")]



    optimizer = optim.SGD([{'params': params_other},
                                 {'params': params_sc, 'lr': args.learning_rate / 100.}],
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_accuracy, test_accuracy = [], []
    train_loss, test_loss = [], []

    best_accuracy = 0

    for epoch in range(1, args.epochs+1):

        loss, accuracy = run(epoch, train_loader, train=True, fp16=args.fp16)
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = run(epoch, test_loader, train=False, fp16=args.fp16)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

        if accuracy > best_accuracy:
            state = {
                'accuracy': accuracy,
                'epoch': epoch,
                'net': net.state_dict()
            }
            torch.save(net.state_dict(), 'models/{}.pth'.format(args.name))
            best_accuracy = accuracy
        scheduler.step()

    plot = Plotting('plots/{}-results.png'.format(args.name))
    plot.draw(train_loss, train_accuracy, test_loss, test_accuracy)
