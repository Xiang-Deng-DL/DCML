from torch.nn import Module
import torch


class RegLoss(Module):
    def __init__(self, instance):
        super(RegLoss, self).__init__()
        
        self.instance = instance

    def forward(self, fea1, fea2, labels1, labels2, batch_w):
        
        #nB = inputs.size(0) # batch_size
        if torch.equal(labels1, labels2):
            difs = torch.sum( (fea1-fea2)**2, 1)
            difs = torch.sum(difs*batch_w)/batch_w.sum()
        else:
            difs = 0.0
        return difs