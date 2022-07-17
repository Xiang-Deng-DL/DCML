from collections import OrderedDict
import torch.nn as nn



class ATL(nn.Module):

    def __init__(self, dim_in=1024, reduce=2):
        super(ATL, self).__init__()
        
        self.fc1 = nn.Linear(in_features=dim_in, out_features=int(dim_in/reduce), bias=False)
        #('bn2', nn.BatchNorm1d(int(dim_in/reduce), eps=1e-5)),
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=int(dim_in/reduce), out_features=dim_in, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        
        shape = input.shape
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        out = x.view(shape[0], shape[1], 1, 1)
        
        return out
