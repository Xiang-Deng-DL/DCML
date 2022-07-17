from torch.nn import Module
import torch


class RegLoss(Module):
    def __init__(self, instance):
        super(RegLoss, self).__init__()
        
        self.instance = instance

    def forward(self, inputs, targets, weights):
        
        target_list = targets.tolist()
        label_set = set(target_list)
        
        label_set = list(label_set)
        
        class_means_dict = dict()
        for label in label_set:
            class_fea = inputs[targets==label]
            class_fea = torch.mean( class_fea, 0)
            class_means_dict[label] = class_fea.detach()
            
        
        means = [class_means_dict[key] for key in target_list]
        means = torch.stack(means, 0)
        
        difs = torch.sum( (inputs-means)**2, 1)
        
        difs = torch.sum(difs*weights)/weights.sum()
        
        return difs