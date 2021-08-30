#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Models.py
@Time    :   2021/08/22 16:18:30
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   Our models.
'''

#%% Import Packages
import torch
import torch.nn as nn

#%% My models
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

    # def set(self):
        # self.train = 

class FNN2(nn.Module):
    def __init__(self):
        super(FNN2, self).__init__()
        self.z = nn.Sequential(
                nn.Linear(28*28, 20),
                nn.ReLU(),
                # nn.Linear(20, 10),
                # nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.z(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(
                in_channels=1,      
                out_channels=16,   
                kernel_size=5,   
                stride=1,          
                padding=2,     
            ),     
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(16, 16, 3, 1, 1), 
            nn.ReLU(),  
            nn.MaxPool2d(2),  
        )
        self.out = nn.Linear(16 * 7 * 7, 10)   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   
        x = self.out(x)
        return x


#%%
if __name__ == '__main__':
    fnn = FNN()
    fnn2 = FNN2()
    cnn = CNN()
    x = torch.ones(5, 1, 28, 28)
    y = fnn(x)
    print(y.shape)
    Y = fnn2(x)
    print(y.shape)
    y = cnn(x)
    print(y.shape)

