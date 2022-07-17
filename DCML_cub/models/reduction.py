from collections import OrderedDict
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class dim_reduce(nn.Module):

    def __init__(self):
        super(dim_reduce, self).__init__()
        
        self.head = nn.Sequential(OrderedDict([
                ('avgpool', nn.AdaptiveAvgPool2d(1)),
                ('bn', nn.BatchNorm2d(1024, eps=1e-5)),
                ('flatten', Flatten()),
                ('fc', nn.Linear(in_features=1024, out_features=128)),
                ]))
  
    def forward(self, bbout):
        
        x = self.head(bbout)

        return x
