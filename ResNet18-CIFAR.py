import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
#import torchvision.models as models
import torchvision.transforms as transforms
import ssl

import models

from matplotlib.ticker import MaxNLocator
from models import ResNet18


class GradientsIterator:

    def __init__(self, model, criterion, data_loader, num_data_iter, paramerters):
        self.model = model
        self.criterion = criterion
        self.data_loader = iter(data_loader)
        self.num_data_iter = num_data_iter
        self.parameters = paramerters
        self.num_iter = 0

    def __iter__(self):
        self.num_iter = 0
        return self

    def __next__(self):
        if self.num_iter >= self.num_data_iter:
            raise StopIteration
        self.num_iter += 1
        inputs, targets = next(self.data_loader)
        batch_size = targets.size(0)

        self.model.zero_grad()
        outputs = self.model(inputs.cuda())
        loss = self.criterion(outputs, targets.cuda())
        loss.backward(create_graph=True)
        grads = self.get_gradients()
        return grads, batch_size

    def get_gradients(self):
        gradients = []
        for parameter in self.parameters:
            gradients.append(0. if parameter.grad is None else parameter.grad + 0.)
        return gradients


class HessianTraceEstimator:

    def __init__(self, model, criterion, data_loader, num_data_points):
        self.model = model
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.num_data_points = num_data_points
        if self.num_data_points >= data_loader.batch_size:
            self.num_data_iter = self.num_data_points // data_loader.batch_size
        else:
            self.num_data_iter = 1
        self.gradients_calculator = GradientsIterator(self.model,
                                                      criterion,
                                                      data_loader,
                                                      self.num_data_iter,
                                                      self.parameters)

    def get_average_traces(self, max_iter=500, tolerance=1e-5):
        trace_last = 0.
        trace_history = []
        trace_average_param = None 

        for i in range(max_iter):
            grad_hessian = self.get_gradients_hessian()
            trace_history.append(grad_hessian)
            trace_average_param = self.get_mean(trace_history)
            trace_tmp = torch.sum(trace_average_param)

            diff_avg = torch.abs((trace_tmp - trace_last) / (trace_last + 1e-6))
            if diff_avg < tolerance:
                break
            trace_last = trace_tmp
        return torch.abs(trace_average_param)

    def sample_rademacher(self, p):
        return torch.randint_like(p, high=2).cuda()*2-1

    def get_gradients_hessian(self):
        v = [self.sample_rademacher(p) for p in self.parameters]
        Hv = [torch.zeros(p.size()).cuda() for p in self.parameters]
        for gradients, batch_size in self.gradients_calculator:
            g = torch.autograd.grad(gradients,
                                    self.parameters,
                                    grad_outputs=v,
                                    only_inputs=True,
                                    retain_graph=False)
            Hv = [Hvi + gi * batch_size for Hvi, gi in zip(Hv, g)]
            
        Hv = [Hvi / float(self.num_data_points) for Hvi in Hv]
        trace_avg_norm = torch.stack([torch.sum(Hvi * vi) / Hvi.size().numel() for (Hvi, vi) in zip(Hv, v)])
        return trace_avg_norm

    @staticmethod
    def get_mean(data):
        return torch.mean(torch.stack(data), dim=0)


def get_cifar10_loaders(train_batch_size=25, test_batch_size=100):

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


def disable_gradients(module):
    if isinstance(module, nn.Sequential) or isinstance(module, models.BasicBlock):
        for layer in module.children():
            disable_gradients(layer)
    elif not isinstance(module, nn.Conv2d) and not isinstance(module, nn.Linear):
        for p in list(module.parameters()):
            if p.requires_grad:
                p.requires_grad = False
    elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
        for p in list(module.parameters()):
            if p.requires_grad:
                p.requires_grad = False
    else:
        layers.append(module)

# Prepare CIFAR10 dataset
train_loader, test_loader = get_cifar10_loaders(train_batch_size=100)

# Prepare the ResNet18 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
model = ResNet18()
model = model.to(device)

# Disable Gradients on BatchNorm Layers
layers = []
for layer in model.children():
    disable_gradients(layer)

print("Proceding for Hessian Trace Estiomation for {} blocks".format(len(layers)))

criterion = nn.CrossEntropyLoss().to(device)
hte = HessianTraceEstimator(model, criterion, train_loader, 10000)
traces = hte.get_average_traces()

# Plot Trace per Layer
fig = plt.figure(tight_layout=True)
plt.yscale('log')
plt.xlabel(r'Blocks$\rightarrow$')
plt.ylabel(r'Average Hessian Trace$\rightarrow$')
traces = traces.detach().cpu().numpy()
x = np.arange(1, len(traces)+1)
plt.plot(x, np.abs(traces), 'o-', color='black')
plt.grid(True, which='both')
plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('ResNet18-CIFAR10-Trace_per_Layer')

# Calculate and Plot Quantization Perturbation
def uniform_quantize(W, bits):
    v = 2**bits
    Wq = torch.round(W * v) / v
    return Wq

def distance(W, bits):
    return torch.abs(W - uniform_quantize(W, bits))

def preference_criterion(net, target=0.2):
    params = sum(p.numel() for p in model.parameters())
    return target * 32 * params / 8 * 1e-6 # in MBs

CLUSTER_SIZE = 2
QUANTS = [2, 4, 8]
PC = preference_criterion(model)

import itertools
fig = plt.figure(tight_layout=True)
arrays = [QUANTS for _ in range(int(len(hte.parameters)/CLUSTER_SIZE))]
config = itertools.product(*arrays)
perturbation = []
sizes = []

# Loop over all the layer's and gather data
for W, t in zip(hte.parameters, traces):
    lp = []
    ls = []
    for b in QUANTS:
        lp.append(np.abs(t)*torch.sum(distance(W, b)))
        ls.append(1e-6*W.numel()*b/8)
    perturbation.append(lp)
    sizes.append(ls)

# Loop over configuration settings
min_omega, min_size, min_cfg = np.inf, np.inf, None
for _, c in enumerate(config):
    size = 0.
    omega = 0.
    bit_allocations = np.repeat(c, CLUSTER_SIZE)
    for x, b in enumerate(bit_allocations):
        y = int(np.log2(b))-1 #index to check
        size += sizes[x][y]
        omega += perturbation[x][y]
    if size < PC and omega < min_omega:
        min_omega = omega
        min_size = size
        min_cfg = bit_allocations
    plt.scatter(size, omega.detach().cpu(), s=1, marker='o', c='black')
plt.scatter(min_size, min_omega.detach().cpu(), s=20, marker='v', c='red')
plt.axvline(x=PC, color='red', linestyle='--')

plt.yscale('log')
plt.ylabel(r'Total Perturbation $\sum_{i=1}^{L}\Omega_i$')
plt.xlabel('Size (MB)')
plt.grid(True, which='both')
plt.savefig('ResNet18-CIFAR10-Perturbation_vs_Size')

print("Pareto Optimal Configuration: {} bit for each layer".format(min_cfg))
