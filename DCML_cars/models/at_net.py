from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .at_linear import ATL
from .at_map import ATM

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AT(nn.Module):

    def __init__(self, at_num = 2):
        super(AT, self).__init__()        
        
        self.at_num = at_num
            
        self.atms = nn.ModuleList()
        for i in range(at_num):
            self.atms.append( ATM(dim_in = 49) )
            
        self.avgpl = nn.AdaptiveAvgPool2d(1)
        
        self.atls = nn.ModuleList()
        for i in range(at_num):
            self.atls.append( ATL(dim_in = 1024, reduce = 2) )

    def forward(self, bbout, train = 1, i = 0):    

        if train==1:
            
            x = torch.mean(bbout, 1)#32*7*7
            m_shape = x.shape
            x = x.view(m_shape[0], -1) #32*49
            
            at_m = self.atms[i](x)
            at_m = at_m.view( m_shape[0], 1, m_shape[1], m_shape[2] )
            out1 = bbout*at_m
                
            z = self.avgpl(out1)
            z = z.view(z.shape[0], -1)
            at_l = self.atls[i](z)
            
            bbout = out1*at_l
            
        else:
            
            x = torch.mean(bbout, 1)
            m_shape = x.shape
            x = x.view(m_shape[0], -1)#32*49
 
            w1_mean = []
            w2_mean = []
            w3_mean = []
            w4_mean = []
            for j in range(self.at_num):
                w1_mean.append( self.atms[j].fc1.weight )
                w2_mean.append( self.atms[j].fc2.weight )
                
                w3_mean.append( self.atls[j].fc1.weight )
                w4_mean.append( self.atls[j].fc2.weight )
                
            w1_mean = torch.stack(w1_mean, 0).mean(0)
            w2_mean = torch.stack(w2_mean, 0).mean(0)
            w3_mean = torch.stack(w3_mean, 0).mean(0)
            w4_mean = torch.stack(w4_mean, 0).mean(0)
            
            x = F.linear(x, w1_mean)
            x = F.relu(x)
            x = F.linear(x, w2_mean)
            x = F.sigmoid(x)
            x = x.view(m_shape[0], 1, m_shape[1], m_shape[2])
            bbout = bbout*x
            
            z = self.avgpl(bbout)
            shape = z.shape
            z = z.view(z.shape[0], -1)
            z = F.linear(z, w3_mean)
            z = F.relu(z)
            z = F.linear(z, w4_mean)
            z = F.sigmoid(z)
            z = z.view(shape[0], shape[1], 1, 1)
            bbout = bbout*z          
        
       
        return bbout
