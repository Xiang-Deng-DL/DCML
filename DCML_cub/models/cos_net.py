from __future__ import print_function

import torch.nn as nn
import math
import torch.nn.functional as F

class CosNet(nn.Module):
    """Linear Layer"""
    def __init__(self, scale = 100.0, margin = 0.6182, dim_in=128, dim_out=75):
        super(CosNet, self).__init__()
        self.dim_out = dim_out
        self.margin = margin
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.scale = scale
    def forward(self, x, label):
        norm_x= x.norm(dim=1, p=2, keepdim=True)
        x = x/norm_x
        norm_w= self.linear.weight.norm(dim=1, p=2, keepdim=True)# 75*512 -> 75
        
        x = self.linear(x)#32*100
        x = x.t() #100*32
        x = x/norm_w #100*32
        x = x.t() #32*100
        one_h = F.one_hot(label, num_classes=self.dim_out) #32*100
        one_h = one_h.float()
        x = self.scale*(x - self.margin*one_h)
        return x