#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fusion.py
@Time    :   2021/08/22 21:57:50
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   
'''

# %% Import Packages
import torch
import torch.nn as nn
import torch.utils.data as Data

# %%
class Hessian_container():
    def __init__(self, parameters, device):
        self.numel = parameters.numel()
        self.shape = parameters.shape
        self.hessian = torch.zeros([self.numel, self.numel]).to(device)

    def add(self, grad):
        self.hessian += torch.mm(grad.T, grad)

    def __repr__(self):
        p = f"[Hessian | numel:{self.numel}, shape:{[int(i) for i in self.shape]}]"
        return p


class Hessian_Matrix():
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
            p = Hessian_container(i, self.device)
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




#%%
def fusion(Tasks, Parm, device=torch.device("cpu"), mode=0):
    Hessian = []
    models = []
    for n, task in enumerate(Tasks):
        print(f'Task {n}')
        d_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=100,
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
                for j in range(y.shape[1]):
                    grad = torch.autograd.grad(y[0, j], model.parameters(), retain_graph=True, create_graph=True)
                    Hessian[-1].add(grad)

        elif mode == 1:
            y = model(data)
            y = y.view(y.shape[0], -1)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    grad = torch.autograd.grad(y[i, j], model.parameters(), retain_graph=True, create_graph=True)
                    Hessian[-1].add(grad)
    print('Finish.')
    return Hessian


# def model_fusion(Hessian, FusionNet):
#     Fusion_Hessian = Hessian_Matrix(FusionNet)
#     for i in range(Fusion_Hessian.blocks_num):

#         Fusion_Hessian.parameters[i] = 

    






#%%
# import torch
# import torch.nn as nn
# import time
# class MM(nn.Module):
#     def __init__(self):
#         super(MM, self).__init__()
#         self.linear1 = nn.Linear(100, 50)
#         self.ReLU = nn.ReLU()
#         self.linear2 = nn.Linear(50, 100)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.ReLU(x)
#         y = self.linear2(x)
#         return y


# model = MM()
# x = torch.randn([1000, 100])

# start = time.time()
# y = model(x)
# save = []



# start = time.time()
# save2 = []
# for i in range(x.shape[0]):
#     y = model(x[i:i+1])
#     for j in range(y.shape[1]):
#         a = torch.autograd.grad(y[0,j], model.parameters(), retain_graph=True, create_graph=True)
       
# print(time.time()-start)
# %%
class A(list):
    def __



# %%
