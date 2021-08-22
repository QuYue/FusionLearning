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
from data_input import data_input, data_split, DATASET
from Models import *
from process import *

# %%
class PARM():
    def __init__(self):
        self.data = DATASET()
        self.dataset_ID = 1
        self.test_size = 0.2
        self.batch_size = 500
        self.epoch = 100
        self.epoch2 = 100
        self.use_device = 'cuda:0' # 'cpu' or 'cuda:0', 'cuda:1'...(GPU ID)
        self.random_seed = 1
        self.model =  FNN
        self.optimizer = torch.optim.SGD
        self.optimizer2 = torch.optim.Adam
        self.lr = 0.1
        self.lr2 = 0.01
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

# %% Read data
datasets = data_input(Parm)
Tasks = []
Train = [[],[]]
Test = [[],[]]
Origin = TASK('origin')
Fusion_task = TASK('fusion')

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
    Tasks.append(task)

    Train[0].extend(data[0][0])
    Train[1].extend(data[0][1])
    Test[0].extend(data[1][0])
    Test[1].extend(data[1][1])
Origin.train = Data.TensorDataset(torch.stack(Train[0]).type(torch.FloatTensor),
                            torch.tensor(Train[1]).type(torch.LongTensor))
Origin.test = Data.TensorDataset(torch.stack(Test[0]).type(torch.FloatTensor), 
                            torch.tensor(Test[1]).type(torch.LongTensor))
Origin.train_loader = Data.DataLoader(dataset=Origin.train,
                                      batch_size=Parm.batch_size,
                                      shuffle=True)
Origin.test_loader = Data.DataLoader(dataset=Origin.test, 
                                     batch_size=1000,
                                     shuffle=False)
Fusion_task.train = Origin.train
Fusion_task.test = Origin.test
Fusion_task.train_loader = Origin.train_loader
Fusion_task.test_loader = Origin.test_loader

#%% Create Models
Model = Parm.model
Origin.model =  Model().to(Parm.device)
Origin.optimizer = Parm.optimizer(Origin.model.parameters(), lr=Parm.lr)
for i in range(Parm.task_number):
    Tasks[i].model = Model().to(Parm.device)
    Tasks[i].optimizer = Parm.optimizer(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_net = Model().to(Parm.device)

# loss function
loss_func = torch.nn.CrossEntropyLoss()

# %% SoloNet Training
start = time.time()
for epoch in range(Parm.epoch):
    print(f"Epoch {epoch}", end=' ')
    for task in Tasks:
        training_process(task, loss_func, Parm)
        testing_process(task, Parm)
    accuracy, name = [], []
    for t, task in enumerate(Tasks):
        accuracy.append(task.test_accuracy[task.ID])
        name.append(f"Task{task.ID}")
        print(f"| task {t}: {task.test_accuracy[task.ID][-1]}", end=' ')
    print('')
finish = time.time()
Parm.time['SoloNet'] = finish - start

# %% Origin Training
print('Origin Training')
name_t = 'Origin'
Origin.model =  Model().to(Parm.device)
Origin.optimizer = Parm.optimizer2(Origin.model.parameters(), lr=Parm.lr2)
Origin.clear()
start = time.time()
for epoch in range(Parm.epoch2):
    print(f"Epoch {epoch}", end=' ')
    training_process(Origin, loss_func, Parm)
    testing_process(Origin, Parm)
    Origin.time.append(time.time()-start)
    print(f"| task {Origin.ID}: {Origin.test_accuracy[Origin.ID][-1]}")
finish = time.time()
Parm.time[name_t] = Origin.time
Parm.result[name_t] = Origin.test_accuracy
print(f"{name_t}: {Parm.time[name_t][-1]}s")
# %%
