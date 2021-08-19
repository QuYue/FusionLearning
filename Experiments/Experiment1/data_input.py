# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   data_input.py
@Time    :   2021/08/08 20:50:54
@Author  :   QuYue
@Email   :   quyue1541@gmail.com
@Desc    :   Input data.
"""
#%% Import Packages
import torch
import torchvision
from sklearn.model_selection import train_test_split

#%% Fusions
def data_input(dataset_no=1, download=True):
    # Input Data
    if dataset_no == 1 or 'Disjoint_MNIST':
        dataset_no = 1
        print('Loading dataset 1： Disjoint_MNIST...')
        data_train, data_test = MNIST_input(download)


def MNIST_input(download=True):
    # Data1 MNIST
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (1.0,))])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                            train=True,
                                            transform=trans,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                           train=False,
                                           transform=trans,
                                           download=download)
    return data_train, data_test