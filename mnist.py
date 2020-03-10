import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from time import time
from torchvision import datasets, transforms
import matplotlib.pylab as plt
from optical_nn import *
import complex_torch_var as ct
from time import time
import os
from default_params import *
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import math
TEST_SIZE = 10_000
BATCH_SIZE = 100
USE_CUDA = False
if USE_CUDA:
    print('Using CUDA')
    DEVICE = th.device('cuda')
else:
    print('Using CPU')
    DEVICE = th.device('cpu')

# Define loader
def mnist_loader(train=True, batch_size=BATCH_SIZE, shuffle=True):
    loader =  th.utils.data.DataLoader(
            datasets.MNIST('../data', train=train, download=True, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=shuffle)
    return loader

# Get X for testing
for data, target in mnist_loader(train=False, batch_size=100, shuffle=False):
    continue
data = data.view(-1, 28**2)
data, target = data.to(DEVICE), target.to(DEVICE)
X0 = data[82][None, :]


# Network dims
N_IN = 28**2//2


def mnist_ONN(unitary=Unitary, num_in=N_IN, num_out=10, num_h1=256, num_h2=256, hidden_units=[256, 256], device=DEVICE, sigma_PS=0, sigma_BS=0, T0=0.03):
    """
    Creates a MLP for training on MNIST
    args:
        unitary: The type of unitary layer used (GridUnitary, FFTUnitary, etc.)
        num_h1: The number of hidden units in the first layer
        num_h2: The number of hidden units in the second layer
        device: The device to be used by torch. 'cpu' or 'cuda'
        sigma_PS: The stdev on uncertainty added to phaseshifter
        sigma_BS: The stdev on uncertainty added to beamsplitter
    returns:
        A th.nn.Sequential module with above features
    """
    f = ShiftedSoftplus(T=T0)
    layers = [
        Linear(num_in, hidden_units[0], sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            Linear(nh_, nh, sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        Linear(hidden_units[-1], num_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS, UNet=unitary),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = NoisySequential(*layers).to(device)
    return net
def mnist_complex(num_in=N_IN, num_out=10, hidden_units=[256, 256], device=DEVICE, sigma=0, T0=0.03):
    print("hello")
    f = ShiftedSoftplus(T=T0)
    layers = [
        ComplexLinear(392, hidden_units[0]),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            ComplexLinear(nh_, nh),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        ComplexLinear(hidden_units[-1], num_out),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = nn.Sequential(*layers).to(device)
    def to_grid_net(rand_S=True):
        grid_net = mnist_ONN(num_in=num_in, num_out=num_out, hidden_units=hidden_units)
        for lc, lo in zip(net, grid_net):
            if isinstance(lc, ComplexLinear):
                assert isinstance(lo, Linear)
                assert lc.weight.shape == (2*lo.D_out, 2*lo.D_in)
                M = lc.weight.to('cpu').data
                t0 = time()
                print(f'Converting weights of size {M.shape}')
                lo.emul_M(M, rand_S=rand_S)
                print(time()-t0)
        return grid_net
    net.to_grid_net = to_grid_net
    return net
def train(model, n_epochs, log_interval, optim_params, batch_size=10, criterion=nn.NLLLoss(), device=DEVICE, epoch_callback=None, log_callback=None):
    loader = mnist_loader(train=True, batch_size=batch_size)
    optimizer = optim.SGD(model.parameters(), **optim_params)
    #criterion = nn.NLLLoss()
    tx = time()
    t0 = time()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0 and batch_idx != 0:
                t = time() - t0
                t0 = time()
                acc = (out.argmax(1) == target).float().mean()
                #acc = get_acc(model, device)
                out = model(data)
                print(f'Epoch: {epoch}, Train loss: {loss.float():.4f}, Train acc: {acc:.4f}, Time/it: {t/log_interval * 1e3:.4f} ms')
                if log_callback:
                    log_callback(model, epoch)
        if epoch_callback:
            epoch_callback(model, epoch)
        print("EpochTime:" + str(time() - tx))
        tx = time()
def get_acc(model, device=DEVICE):
    confusion = None
    with th.no_grad():
        for data, target in mnist_loader(train=False, batch_size=TEST_SIZE):
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(1)
            acc = (pred == target).float().mean()
            confusion = confusion_matrix(target.cpu(), pred.cpu())
    Y_pred = model(data)
    out = Y_pred.data.max(1, keepdim=True)[1]
    Y_pred = torch.exp(Y_pred)
    Y_pred.detach()
    Y_real = torch.zeros((10000, 10), requires_grad = False)
    Y_pred = Y_pred.detach().numpy()
    for i in range(10000):
        Y_real[i][target[i]] = 1
    # Plot linewidth.
    Y_real = Y_real.numpy()
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        temp = []
        fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print("hello")
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_real.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 10

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot micro/macro ROC curves
    # plt.figure(1)
    # plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #             ''.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'
    #             ''.format(roc_auc["macro"]),
    #         color='navy', linestyle=':', linewidth=4)

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic for multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    # Plot class ROC curves
    plt.figure(2)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','deeppink','navy','yellow','purple','gray','red','green','brown','blue'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.6f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    plt.show()
    return acc.item(), confusion

class StackedFFTUnitary(nn.Sequential):
    def __init__(self, D, n_stack=None, sigma_PS=0, sigma_BS=0):
        if n_stack is None:
            P = int(log2(D))
            n_stack = int(D//P)
        layers = [FFTUnitary(D, sigma_PS=sigma_PS, sigma_BS=sigma_BS) for _ in range(n_stack)]

        super().__init__(*layers)
        self.sigma_PS = sigma_PS
        self.sigma_BS = sigma_BS
    @property
    def sigma_PS(self):
        return self._sigma_PS
    @property
    def sigma_BS(self):
        return self._sigma_BS
    @sigma_PS.setter
    def sigma_PS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_PS = new_sig
        self._sigma_PS = new_sig
    @sigma_BS.setter
    def sigma_BS(self, new_sig):
        # Updates sigma of all layers
        for layer in self:
            layer.sigma_BS = new_sig
        self._sigma_BS = new_sig

def mnist_stacked_fft(n_stack, device=DEVICE, T0=0.03):
    f = ShiftedSoftplus(T=0.03)
    layers = [
            Linear(N_IN, 256),
            ModNonlinearity(f=f),
            StackedFFTUnitary(256, n_stack=n_stack),
            Diagonal(256, 256),
            StackedFFTUnitary(256, n_stack=n_stack),
            ModNonlinearity(f=f),
            Linear(256, 10),
            ComplexNorm(),
            nn.LogSoftmax(dim=1)
            ]
    net = NoisySequential(*layers).to(device)
    return net

def mnist_grid_truncated(num_in=N_IN, num_out=10, hidden_units=[256, 256], device=DEVICE, sigma_PS=0, sigma_BS=0, T0=0.03):
    f = ShiftedSoftplus(T=T0)
    layers = [
        TruncatedGridLinear(num_in, hidden_units[0], sigma_PS=sigma_PS, sigma_BS=sigma_BS),
        ModNonlinearity(f=f)
            ]
    for nh_, nh in zip(hidden_units[:-1], hidden_units[1:]):
        layers.extend([
            TruncatedGridLinear(nh_, nh, sigma_PS=sigma_PS, sigma_BS=sigma_BS),
            ModNonlinearity(f=f),
                ])
    layers.extend([
        TruncatedGridLinear(hidden_units[-1], num_out, sigma_PS=sigma_PS, sigma_BS=sigma_BS),
        ComplexNorm(),
        nn.LogSoftmax(dim=1)
        ])
    net = NoisySequential(*layers).to(device)
    return net


if __name__ == '__main__':

    LR_FFT = 5e-2
    LR_GRID = 2.5e-4
    LR_COMPLEX = 5e-3
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 1
    train_params['batch_size'] = 100

    optim_params = {}
    optim_params['lr'] = LR_FFT
    optim_params['momentum'] = .9

    net = mnist_ONN(unitary = FFTUnitary)
    print(net(X0))
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    th.save(net, os.path.join(DIR_TRAINED_MODELS, 'truncated_grid.pth'))
    assert False
    net = mnist_stacked_fft(1)
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    th.save(net, os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_1.pth'))
    print(get_acc(net))
    net_loaded = th.load(os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_1.pth'))
    print(get_acc(net_loaded))

    assert False
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    th.save(net, os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_32.pth'))
    print(get_acc(net))
    net_loaded = th.load(os.path.join(DIR_TRAINED_MODELS, 'stacked_fft_32.pth'))
    print(get_acc(net_loaded))
