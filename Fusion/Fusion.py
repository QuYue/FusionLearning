#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fusion.py
@Time    :   2021/08/22 21:57:50
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   Fusion
'''

# %% Import Packages
import torch
import torch.nn as nn
import torch.utils.data as Data
from functools import reduce

# %% 
class Grad(list):
    # Gradient
    def reshape(self):
        for i in range(len(self)):
            self[i] = self[i].data.view(1, -1)
    
    @property
    def shape(self):
        return [i.shape for i in self]

    def stack(self, new_Grad):
        assert len(self) == len(new_Grad)
        g = [[]] * len(self)
        for i in range(len(self)):
            g[i] = torch.vstack([self[i], new_Grad[i]])
        return Grad(g)

    def assign(self, new_list):
        g = [[]] * len(new_list)
        for i in range(len(new_list)):
            g[i] = torch.hstack([self[h] for h in new_list[i]])
        return Grad(g)
        

class HessianMatrix():
    # Hessian Matrix
    def __init__(self, parameters, device):
        self.numel = parameters.numel()
        self.shape = parameters.shape
        self.hessian = torch.zeros([self.numel, self.numel]).to(device)

    def add(self, grad):
        self.hessian += torch.mm(grad.T, grad)

    def __repr__(self):
        p = f"{{Hessian |numel:{self.numel}|shape:{[int(i) for i in self.shape]}}}"
        return p


class HessianMatrices(list):
    # Hessian Matrix
    def add_grad(self, grad):
        for i in range(len(self)):
            self[i].add(grad[i])


class Material():
    def __init__(self, model):
        self.model = model
        self.parameters = list(self.model.parameters())
        self.device = self.parameters[0].device
        self.count = 0
        self.blocks_num = len(self.parameters)
        self.create_Hessian_list()

    def create_Hessian_list(self):
        Hessian_list = []
        for i in self.parameters:
            p = HessianMatrix(i, self.device)
            Hessian_list.append(p)
        self.Hessian_list = HessianMatrices(Hessian_list)
        
    def add_grad(self, grad):
        self.count +=  grad[0].shape[0]
        self.Hessian_list.add_grad(grad)

    def update_block(self, new_parameters, block_id):
        block = self.parameters[block_id]
        assert new_parameters.numel() == block.numel()
        block.data = new_parameters.view(block.shape)
    
    def __repr__(self):
        p = f"Material with {self.count} data"
        return p

#%%
def fusion_prepare(Tasks, Parm, device=torch.device("cpu"), mode=0):
    fusion_materials = []
    models = []
    for n, task in enumerate(Tasks):
        print(f'Task {n}')
        d_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=Parm.fusion.samples,
                                        shuffle=True)
        for data, _ in d_loader:
            data = data.to(device)
            break
        model = task.model.to(device)
        models.append(model)

        fusion_materials.append(Material(model))
        if mode == 0:
            for i in range(data.shape[0]):
                d = data[i:i+1]
                y = model(d)
                y = y.view(1, -1)
                grad = []
                for j in range(y.shape[1]):
                    grad.append(Grad(torch.autograd.grad(y[0, j], model.parameters(), retain_graph=True, create_graph=True)))
                    grad[-1].reshape()
                grad = reduce(lambda x,y: x.stack(y), grad)
                fusion_materials[-1].add_grad(grad)

        elif mode == 1:
            y = model(data)
            y = y.view(y.shape[0], -1)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    grad = Grad(torch.autograd.grad(y[i, j], model.parameters(), retain_graph=True, create_graph=True))
                    grad.reshape()
                    fusion_materials[-1].add_grad(grad)
        elif mode == 2:
            for i in range(data.shape[0]):
                d = data[i:i+1]
                y = model(d)
                y = y.view(1, -1)
                grad = []
                for j in range(y.shape[1]):
                    grad = Grad(torch.autograd.grad(y[0, j], model.parameters(), retain_graph=True, create_graph=True)) 
                    grad.reshape()
                    fusion_materials[-1].add_grad(grad)
    print('Finish.')
    return fusion_materials


def fusion(fusion_materials, FusionNet, Parm):
    fusioned = Material(FusionNet)
    blocks_num = fusioned.blocks_num
    for layer in range(blocks_num):
        print(f"Layer {layer} | {blocks_num}")
        H = torch.zeros_like(fusioned.Hessian_list[layer].hessian)
        Htheta = torch.zeros(H.shape[0],1).to(H.device)
        new_parameters = torch.zeros_like(fusioned.parameters[layer])
        for i in range(len(fusion_materials)):
            H += fusion_materials[i].Hessian_list[layer].hessian / Parm.fusion.samples
            Htheta  += torch.mm(fusion_materials[i].Hessian_list[layer].hessian / Parm.fusion.samples, 
                                fusion_materials[i].parameters[layer].data.view(-1,1))
        H_inverse = H.pinverse(rcond=Parm.fusion.pinvrcond)
        new_parameters = torch.mm(H_inverse, Htheta)
        fusioned.update_block(new_parameters, layer)
    return fusioned
            



        
#%%

            



        








