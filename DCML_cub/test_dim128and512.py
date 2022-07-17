
import os
import torch

import numpy as np
from models.bninception import BNInception
from models.at_net import AT
from models.reduction import dim_reduce


from torch.utils.data import DataLoader

from evaluation import pairwise_similarity, Recall_at_ks, MAP_R, R_Precision
from data_engine import MSBaseDataSet
import copy
from config import get_config

def test_dim128(model, at, head, conf):

    model.eval()
    at.eval()
    head.eval()
    
    if conf.use_dataset == 'CUB':
        dataset = MSBaseDataSet(conf, '/home/xiang/data/cub-200-2011_metric/CUB_200_2011/cub_test.txt', 
                                transform=conf.transform_dict['center-crop'], mode='RGB')
        
        loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)
        
        loader_enum = enumerate(loader)
        feas = torch.tensor([])
        labels = np.array([])
        with torch.no_grad():
            while True:
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    break
               
                imgs = data['image'].cuda()
                label = data['label'].cuda()
                
                output1 = model(imgs)                   
                output1 = at(output1, train = 0)                   
                output1 = head(output1)
                    
                norm = output1.norm(dim=1, p=2, keepdim=True)
                output1 = output1.div(norm.expand_as(output1))
                feas = torch.cat((feas, output1.cpu()), 0)
                labels = np.append(labels, label.cpu().numpy())
                
        sim_mat = pairwise_similarity(feas)
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
        sim_mat2 = copy.deepcopy(sim_mat)
            
        sim_mat3 = copy.deepcopy(sim_mat)
            
        recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)
            
        mapr = MAP_R(sim_mat2, labels)
            
        rp = R_Precision(sim_mat3, labels)

    return recall_ks, mapr, rp

def test_dim512( model_list, at_list, head_list, conf):
    
    for model in model_list:
        model.eval()
    for at in at_list:
        at.eval()   
    for head in head_list:
        head.eval()

    if conf.use_dataset == 'CUB':
        dataset = MSBaseDataSet(conf, '/home/xiang/data/cub-200-2011_metric/CUB_200_2011/cub_test.txt', 
                                transform=conf.transform_dict['center-crop'], mode='RGB')
        
        loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)
        
        loader_enum = enumerate(loader)
        feas = torch.tensor([])
        labels = np.array([])
        with torch.no_grad():
            while True:
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    break
               
                imgs = data['image'].cuda()
                label = data['label'].cuda()
                
                output1 = []
                for i in range( len(model_list) ):
                    f = model_list[i](imgs)
                    f = at_list[i](f, train = 0)
                    f = head_list[i](f)
                    output1.append( f )
                    
                output1 =  torch.cat(output1,1)
                norm = output1.norm(dim=1, p=2, keepdim=True)
                output1 = output1.div(norm.expand_as(output1))
                feas = torch.cat((feas, output1.cpu()), 0)
                labels = np.append(labels, label.cpu().numpy())
        
        _, dim = feas.shape
        assert dim==512

        sim_mat = pairwise_similarity(feas)
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
        sim_mat2 = copy.deepcopy(sim_mat)
            
        sim_mat3 = copy.deepcopy(sim_mat)
            
        recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)
            
        mapr = MAP_R(sim_mat2, labels)
            
        rp = R_Precision(sim_mat3, labels)
        
    return recall_ks, mapr, rp

if __name__ == '__main__':
    
    conf = get_config()
    
    if conf.test_pretrained_model!=1:
        
        model1 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_atnum{atnum}_regcof{regcof}_atcof{atcof}_wcof{w_cof}_maxw{maxw}max_at_cof{maxat_cof}.pth'.format(
                cross_id=1, seed=conf.seed, atnum=conf.at_num, regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof, maxw=conf.maxw_cof, maxat_cof=conf.maxat_cof))  
        model1.load_state_dict(torch.load(model_path)['model'])
        at1 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at1.load_state_dict(torch.load(model_path)['at'])
        head1 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head1.load_state_dict(torch.load(model_path)['head'])
    
        model2 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_atnum{atnum}_regcof{regcof}_atcof{atcof}_wcof{w_cof}_maxw{maxw}max_at_cof{maxat_cof}.pth'.format(
                cross_id=2, seed=conf.seed+2, atnum=conf.at_num, regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof, maxw=conf.maxw_cof, maxat_cof=conf.maxat_cof))
        model2.load_state_dict(torch.load(model_path)['model'])
        at2 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at2.load_state_dict(torch.load(model_path)['at'])
        head2 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head2.load_state_dict(torch.load(model_path)['head']) 
    
    
        model3 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_atnum{atnum}_regcof{regcof}_atcof{atcof}_wcof{w_cof}_maxw{maxw}max_at_cof{maxat_cof}.pth'.format(
                cross_id=3, seed=conf.seed+4, atnum=conf.at_num, regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof, maxw=conf.maxw_cof, maxat_cof=conf.maxat_cof))
        model3.load_state_dict(torch.load(model_path)['model'])
        at3 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at3.load_state_dict(torch.load(model_path)['at'])
        head3 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head3.load_state_dict(torch.load(model_path)['head'])


        model4 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_atnum{atnum}_regcof{regcof}_atcof{atcof}_wcof{w_cof}_maxw{maxw}max_at_cof{maxat_cof}.pth'.format(
                cross_id=4, seed=conf.seed+6, atnum=conf.at_num, regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof, maxw=conf.maxw_cof, maxat_cof=conf.maxat_cof))
        model4.load_state_dict(torch.load(model_path)['model'])
        at4 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at4.load_state_dict(torch.load(model_path)['at'])
        head4 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head4.load_state_dict(torch.load(model_path)['head'])

        recall_ks, mapr, rp = test_dim128(model1, at1, head1, conf)
        print(f'test model1 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                     
        recall_ks, mapr, rp = test_dim128(model2, at2, head2, conf)
        print(f'test model2 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')

        recall_ks, mapr, rp = test_dim128(model3, at3, head3, conf)
        print(f'test model3 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')

        recall_ks, mapr, rp = test_dim128(model4, at4, head4, conf)
        print(f'test model4 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                              
        recall_ks, mapr, rp = test_dim512([model1, model2, model3, model4], [at1, at2, at3, at4], [head1, head2, head3, head4], conf)
        print(f'test model dim 512: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
        
    else:#load and test our pretrained models

        model1 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = './pretrained_models/model1.pth'
        model1.load_state_dict(torch.load(model_path)['model'])
        at1 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at1.load_state_dict(torch.load(model_path)['at'])
        head1 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head1.load_state_dict(torch.load(model_path)['head'])
    
    
        model2 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = './pretrained_models/model2.pth'
        model2.load_state_dict(torch.load(model_path)['model'])
        at2 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at2.load_state_dict(torch.load(model_path)['at'])
        head2 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head2.load_state_dict(torch.load(model_path)['head']) 
    
    
        model3 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = './pretrained_models/model3.pth'
        model3.load_state_dict(torch.load(model_path)['model'])
        at3 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at3.load_state_dict(torch.load(model_path)['at'])
        head3 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head3.load_state_dict(torch.load(model_path)['head'])


        model4 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = './pretrained_models/model4.pth'
        model4.load_state_dict(torch.load(model_path)['model'])
        at4 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at4.load_state_dict(torch.load(model_path)['at'])
        head4 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head4.load_state_dict(torch.load(model_path)['head'])

    
        recall_ks, mapr, rp = test_dim128(model1, at1, head1, conf)
        print(f'test model1 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                     
        recall_ks, mapr, rp = test_dim128(model2, at2, head2, conf)
        print(f'test model2 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')

        recall_ks, mapr, rp = test_dim128(model3, at3, head3, conf)
        print(f'test model3 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')

        recall_ks, mapr, rp = test_dim128(model4, at4, head4, conf)
        print(f'test model4 dim128: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                              
        recall_ks, mapr, rp = test_dim512([model1, model2, model3, model4], [at1, at2, at3, at4], [head1, head2, head3, head4], conf)
        print(f'test model dim 512: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
 
        
    
