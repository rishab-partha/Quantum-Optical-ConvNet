'''
This file contains functionalities for training and loading the various models that were designed in this project.
Based on the codebase at https://github.com/mike-fang/imprecise_optical_neural_network. The majority of the file is my own code.

@version 3.8.2021
'''


import numpy as np
print(1)
import torch as th
print(2)
import matplotlib.pylab as plt
print(3)
from optical_nn import *
print(4)
import complex_torch_var as ct
print(5)
from mnist import *
print(6)
import os
print(7)
from time import time
print(8)
from functools import partial
print(9)
from glob import glob
print(10)
from default_params import *
print(11)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/complexnet')
print("writer created")


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Good learning rates for different networks
LR_FFT = 5e-2
LR_GRID = 2.5e-4
LR_COMPLEX = 5e-3

'''
Train networks based on ComplexNet for 10 epochs.

Inputs:
    f: location to save the network
    n_h: # of hidden units
'''
def train_complex(f=F_COMPLEX_TRAIN, n_h=[256, 256]):

    # Define training parameters
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 10
    train_params['batch_size'] = 100

    # Define optimization parameters
    optim_params = {}
    optim_params['lr'] = 1.0e-3
    optim_params['momentum'] = .9

    # Create model
    net = mnist_complex(hidden_units=n_h)
    print(net)

    # Train for 10 epochs, slashing learning rate after 5
    train(net, **train_params, optim_params=optim_params, writer=writer)
    optim_params['lr'] /= 5

    train(net, **train_params, optim_params=optim_params, writer=writer, iteration = 1)
    acc = get_acc(net)

    print(f'Trained ComplexNet with accuracy {acc}.')
    writer.close()
    # Save model
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')

'''
Train networks based on a modified GridNet for 10 epochs.

Inputs:
    f: location to save the network
    n_h: # of hidden units
'''
def train_cgrd(f=F_CGRD_TRAIN):

    # Define training parameters
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    # Define optimization parameters
    optim_params = {}
    optim_params['lr'] = LR_GRID
    optim_params['momentum'] = .9

    # Create model
    net = mnist_ONN(unitary=CGRDUnitary)

    # Train for 10 epochs, slashing learning rate after 5
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)

    print(f'Trained ComplexGridNet with accuracy {acc}.')

    # Save model
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')

'''
Train networks based on GridNet for 10 epochs.

Inputs:
    f: location to save the network
    n_h: # of hidden units
'''
def train_grid(f=F_GRID_TRAIN, n_h=[256, 256]):

    # Define training parameters
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    # Define optimization parameters
    optim_params = {}
    optim_params['lr'] = LR_GRID
    optim_params['momentum'] = .9

    # Create model
    net = mnist_ONN(hidden_units=n_h)

    # Train for 10 epochs, slashing learning rate after 5
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)

    print(f'Trained GridNet with accuracy {acc}.')

    # Save model
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')

'''
Train networks based on FFTNet for 10 epochs.

Inputs:
    f: location to save the network
    n_h: # of hidden units
'''
def train_fft(f=F_FFT_TRAIN, n_h=[256, 256]):
    
    # Define training parameters
    train_params = {}
    train_params['n_epochs'] = 5
    train_params['log_interval'] = 100
    train_params['batch_size'] = 100

    # Define optimization parameters
    optim_params = {}
    optim_params['lr'] = LR_FFT*3
    optim_params['momentum'] = .9

    # Create model
    net = mnist_ONN(FFTUnitary, hidden_units=n_h)

    # Train for 10 epochs, slashing learning rate after 5
    train(net, **train_params, optim_params=optim_params)
    optim_params['lr'] /= 5
    train(net, **train_params, optim_params=optim_params)
    acc = get_acc(net)

    print(f'Trained FFTNet with accuracy {acc}.')

    # Save model
    if f:
        th.save(net.state_dict(), f)
        print(f'Saved model to {f}.')

'''
Converts a ComplexNet into a GridNet.

Inputs:
    complex_net: the ComplexNet to convert
    f: location to save the GridNet
    rand_S: randomize GridNet structure
'''
'''
def convert_save_grid_net(complex_net=None, f=None, rand_S=True):
    if complex_net is None:
        complex_net = load_complex()

    if f is None:
        f = F_GRID_TRAIN if rand_S else F_GRID_ORD_TRAIN

    grid_net = complex_net.to_grid_net(rand_S=rand_S).to(DEVICE)
    acc = get_acc(grid_net)
    print(f'Converted to GridNet with accuracy {acc} with {"shuffled" if rand_S else "ordered"} singular values.')
    th.save(grid_net.state_dict(), f)
    print(f'Saved GridNet at {f}')
'''

'''
Train the ComplexNet in batches.

Inputs:
    n_train: Number of batches to train for
    dir: directory to save batches
'''
def batch_train_complex(n_train, dir = DIR_COMPLEX_TRAIN):
    for _ in range(n_train):
        f = os.path.join(dir, f'{time():.0f}')
        train_complex(f=f)

'''
Convert a batch trained ComplexNet to a GridNet

Inputs:
    dir: directory of batches
'''
'''
def batch_convert(dir = DIR_COMPLEX_TRAIN):
    for f in glob(os.path.join(dir, '*')):
        net = load_complex(f)
        convert_save_grid_net(net, f=f+'_grid')
'''

'''
Load a ComplexNet from Directory

Inputs:
    f: Directory of the model

Outputs:
    The loaded model
'''
def load_complex(f=F_COMPLEX_TRAIN):
    net = mnist_complex()
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc, confusion_matrix = get_acc(net)
    print(f'ComplexNet loaded from {f} with accuracy {acc}.')
    print(confusion_matrix)
    return net.to(DEVICE)

'''
Load a GridNet from Directory and generate accuracy/confusion matrices.

Inputs:
    f: Directory of the model
    rand_S: whether or not to randomize GridNet states
    report_acc: whether or not to generate accuracy/confusion matrices

Outputs:
    The loaded model
'''
def load_grid(f=os.path.join(DIR_TRAINED_MODELS, 'grid_1_layer.pth'), rand_S=True, report_acc=True):
    if f is None:
        f = F_GRID_TRAIN if rand_S else F_GRID_ORD_TRAIN
    net = mnist_ONN()
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc, confusion_matrix = get_acc(net)
    print(f'GridNetOrdered loaded from {f} with accuracy {acc}.')
    print(confusion_matrix)
    return net.to(DEVICE)

'''
Load a FFTNet from Directory.

Inputs:
    f: Directory of the model

Outputs:
    The loaded model
'''
def load_fft(f=os.path.join(DIR_TRAINED_MODELS, 'fft_net.pth')):
    net = mnist_ONN(FFTUnitary)
    print(net)
    print(th.load(f, map_location=DEVICE))
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc, confusion_matrix = get_acc(net)
    print(f'FFTNet loaded from {f} with accuracy {acc}.')
    print(confusion_matrix)
    return net.to(DEVICE)

'''
Load a CGRDNet from Directory

Inputs:
    f: Directory of the model

Outputs:
    The loaded model
'''
def load_cgrd(f=F_CGRD_TRAIN):
    net = mnist_ONN(CGRDUnitary)
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc, confusion_matrix = get_acc(net)
    print(f'CGRDNet loaded from {f} with accuracy {acc}.')
    print(confusion_matrix)
    return net.to(DEVICE)

'''
Load a Truncated GridNet from Directory

Inputs:
    f: Directory of the model

Outputs:
    The loaded model
'''
def load_trunc_grid(f=os.path.join(DIR_TRAINED_MODELS, 'truncated_grid.pth')):
    net = mnist_grid_truncated()
    print(net)
    print(th.load(f, map_location=DEVICE))
    net.load_state_dict(th.load(f, map_location=DEVICE))
    acc, confusion_matrix = get_acc(net)
    print(f'Truncated GridNet loaded from {f} with accuracy {acc}.')
    print(confusion_matrix)
    return net.to(DEVICE)

if __name__ == '__main__':
    train_complex()
    net = load_complex()
    
    for data, target in mnist_loader(train=False, batch_size=100, shuffle=False):
        continue
    data = data.view(-1, 28**2)
    data, target = data.to(DEVICE), target.to(DEVICE)
    print(th.max(net(data), dim=1))
