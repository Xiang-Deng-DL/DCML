from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ATM(nn.Module):

    def __init__(self, dim_in = 49):
        super(ATM, self).__init__()
        
        #('bn1', nn.BatchNorm1d(dim_in, eps=1e-5)),
        #('relu1', nn.ReLU(inplace = True)),
        self.fc1 = nn.Linear(in_features=dim_in, out_features=dim_in, bias=False)
        #('bn2', nn.BatchNorm1d(dim_in, eps=1e-5)),
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim_in, out_features=dim_in, bias=False)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        
        shape = input.shape
        
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        out = x.view(shape)
        
        return out
