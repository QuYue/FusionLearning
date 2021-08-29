#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/08/22 15:02:00
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   main function. 
'''

#%% Import Packages
import torch
import numpy as np
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
from data_input import data_input, data_split, DataSetConfig
from Models import *
from process import *
import sys
sys.path.append('../..') # add the path which includes the packages
from Fusion import Fusion

# %%
class PARM():
    def __init__(self):
        self.data = DataSetConfig()
        self.dataset_ID = 1
        self.test_size = 0.2
        self.batch_size = 5000
        self.epoch = 10
        self.epoch2 = 20
        self.use_device = 'cuda:0' # 'cpu' or 'cuda:0', 'cuda:1'...(GPU ID)
        self.random_seed = 1
        self.model =  FNN
        self.optimizer = torch.optim.SGD
        self.optimizer2 = torch.optim.Adam
        self.lr = 0.1
        self.lr2 = 0.001
        self.time = dict()
        self.result = {'SoloNet':{}, 'FusionNet':{}, 'Origin':{}}

    @property
    def dataset_name(self):
        return self.data.data_dict[self.dataset_ID]  
    @property
    def task_number(self):
        return self.data.tasks_num[self.dataset_name]            

Parm = PARM()

# %%
if Parm.use_device !='cpu' and torch.cuda.is_available():
    Parm.device = torch.device(Parm.use_device)
    if torch.cuda.device_count() <= Parm.device.index:
        Parm.device = torch.device('cuda:0')
    Parm.cuda = True
    print(f'Using GPU {Parm.device.index}.')
else:
    Parm.device = torch.device('cpu')
    Parm.cuda = False
    print('Using CPU.')

torch.manual_seed(Parm.random_seed)
torch.cuda.manual_seed(Parm.random_seed)

#%% Task
class TASK:
    def __init__(self, ID=0):
        self.ID = ID
        self.train = []
        self.test = []
        self.train_accuracy = {self.ID:[]}
        self.test_accuracy = {self.ID:[]}
        self.time = []
    def clear(self):  
        self.train_accuracy = {self.ID:[]}
        self.test_accuracy = {self.ID:[]}
        self.time = []

class TData:
    def __init__(self):
        self.x = []
        self.y = []

class DataSET:
    def __init__(self):
        self.train = TData()
        self.test = TData()

# %% Read data
datasets = data_input(Parm)
# Tasks
Origin_task = TASK('origin') # Original task
Fusion_task = TASK('fusion') # FusionNet task
Solo_tasks = []              # SoloNet task
# Data
DataSet = DataSET()

for i in range(Parm.task_number):
    task = TASK(i)
    data = data_split(datasets[i]['data'], datasets[i]['target'], 
                    Parm.test_size, random_state=Parm.random_seed)
    task.train = Data.TensorDataset(torch.stack(data[0][0]).type(torch.FloatTensor), 
                                torch.tensor(data[0][1]).type(torch.LongTensor))
    task.test = Data.TensorDataset(torch.stack(data[1][0]).type(torch.FloatTensor), 
                                torch.tensor(data[1][1]).type(torch.LongTensor))
    task.train_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=Parm.batch_size,
                                        shuffle=True)
    task.test_loader = Data.DataLoader(dataset=task.test, 
                                       batch_size=1000,
                                       shuffle=False)
    Solo_tasks.append(task)

    DataSet.train.x.extend(data[0][0]) # train data x
    DataSet.train.y.extend(data[0][1]) # train data y
    DataSet.test.x.extend(data[1][0])  # test data x
    DataSet.test.y.extend(data[1][1])  # test data y

Origin_task.train = Data.TensorDataset(torch.stack(DataSet.train.x).type(torch.FloatTensor),
                            torch.tensor(DataSet.train.y).type(torch.LongTensor))
Origin_task.test = Data.TensorDataset(torch.stack(DataSet.test.x).type(torch.FloatTensor), 
                            torch.tensor(DataSet.test.y).type(torch.LongTensor))
Origin_task.train_loader = Data.DataLoader(dataset=Origin_task.train,
                                      batch_size=Parm.batch_size,
                                      shuffle=True)
Origin_task.test_loader = Data.DataLoader(dataset=Origin_task.test, 
                                     batch_size=1000,
                                     shuffle=False)
Fusion_task.train = Origin_task.train
Fusion_task.test = Origin_task.test
Fusion_task.train_loader = Origin_task.train_loader
Fusion_task.test_loader = Origin_task.test_loader

#%% Preparation
# Create Models
Model = Parm.model
Origin_task.model =  Model().to(Parm.device)
Origin_task.optimizer = Parm.optimizer(Origin_task.model.parameters(), lr=Parm.lr)
FusionNet = Model().to(Parm.device)
for i in range(Parm.task_number):
    Solo_tasks[i].model = Model().to(Parm.device)
    Solo_tasks[i].optimizer = Parm.optimizer(Solo_tasks[i].model.parameters(), lr=Parm.lr)

# loss function
loss_func = torch.nn.CrossEntropyLoss()

# %% Origin Training
print('Origin Training')
name_t = 'Origin'
Origin_task.model =  Model().to(Parm.device)
Origin_task.optimizer = Parm.optimizer2(Origin_task.model.parameters(), lr=Parm.lr2)
Origin_task.clear()
start = time.time()
for epoch in range(Parm.epoch2):
    print(f"Epoch {epoch}", end=' ')
    training_process(Origin_task, loss_func, Parm)
    testing_process(Origin_task, Parm)
    Origin_task.time.append(time.time()-start)
    print(f"| task {Origin_task.ID}: {Origin_task.test_accuracy[Origin_task.ID][-1]}")
finish = time.time()
Parm.time[name_t] = Origin_task.time
Parm.result[name_t] = Origin_task.test_accuracy
print(f"{name_t}: {Parm.time[name_t][-1]}s")


# %% SoloNet Training
start = time.time()
for epoch in range(Parm.epoch):
    print(f"Epoch {epoch}", end=' ')
    for task in Solo_tasks:
        training_process(task, loss_func, Parm)
        testing_process(task, Parm)
    accuracy, name = [], []
    for t, task in enumerate(Solo_tasks):
        accuracy.append(task.test_accuracy[task.ID])
        name.append(f"Task{task.ID}")
        print(f"| task {t}: {task.test_accuracy[task.ID][-1]}", end=' ')
    print('')
finish = time.time()
Parm.time['SoloNet'] = finish - start


# %%
class Grad():
    def __init__(self, parameters, device):
        self.numel = parameters.numel()
        self.shape = parameters.shape
        self.hessian = torch.zeros([self.numel, self.numel]).to(device)

class HessianMatrix():
    def __init__(self, parameters, device):
        self.numel = parameters.numel()
        self.shape = parameters.shape
        self.hessian = torch.zeros([self.numel, self.numel]).to(device)

    def add(self, grad):
        self.hessian += torch.mm(grad.T, grad)

    def __repr__(self):
        p = f"{{Hessian | numel:{self.numel}, shape:{[int(i) for i in self.shape]}}|"
        return p

class HessianMatrices(list):
    def stack(self, new_HessianMatrix):
        assert self.__len__() == len(new_HessianMatrix)
        for i in range(self.__len__()):
            i = torch.stack([self[i], new_HessianMatrix[i]])




class Material():
    def __init__(self, model):
        self.model = model
        self.parameters = list(self.model.parameters())
        self.device = self.parameters[0].device
        self.count = 0
        self.blocks_num = len(self.parameters)
        self.create_Hessian_list()

    def create_Hessian_list(self):
        self.Hessian_list = []
        for i in self.parameters:
            p = HessianMatrix(i, self.device)
            self.Hessian_list.append(p)
        
    def add(self, grad):
        self.count += 1
        for i in range(len(grad)):
            g = grad[i].data.view(1, -1)
            self.Hessian_list[i].add(g)

    def update_block(self, new_parameters, block_id):
        block = self.parameters[block_id]
        assert new_parameters.numel() == block.numel
        block = new_parameters.view(block.shape)



def fusion(Tasks, Parm, device=torch.device("cpu"), mode=0):
    Hessian = []
    models = []
    for n, task in enumerate(Tasks):
        print(f'Task {n}')
        d_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=1000,
                                        shuffle=True)
        for data, _ in d_loader:
            data = data.to(device)
            break
        model = task.model.to(device)
        models.append(model)

        Hessian.append(Hessian_Matrix(model))
        if mode == 0:
            for i in range(data.shape[0]):
                d = data[i:i+1]
                y = model(d)
                y = y.view(1, -1)
                grad = []
                for j in range(y.shape[1]):
                    grad.append(torch.autograd.grad(y[0, j], model.parameters(), retain_graph=True, create_graph=True))
                
                Hessian[-1].add(grad)

        elif mode == 1:
            y = model(data)
            y = y.view(y.shape[0], -1)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    grad = torch.autograd.grad(y[i, j], model.parameters(), retain_graph=True, create_graph=True)
                    # Hessian[-1].add(grad)
    print('Finish.')
    return Hessian

#%%
start = time.time()
H = fusion(Solo_tasks, Parm, Parm.device, mode=0)
print(time.time()-start)

# %%
d.stack(d)