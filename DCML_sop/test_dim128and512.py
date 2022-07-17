
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

if __name__ == '__main__':

    def test_dim128(model, at, head, conf):
        
        model.eval()
        at.eval()
        head.eval()
       
        if conf.use_dataset=='SOP':

            if conf.use_dataset == 'SOP':
                dataset = MSBaseDataSet(conf, conf.datapth+'sop_test.txt', transform=conf.transform_dict['center-crop'], mode='RGB')

            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=False, pin_memory=True, drop_last=False)

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
                    
            feas = feas.cpu().numpy()
            
            sim_mat = np.matmul( feas, np.transpose(feas) )
 
            np.fill_diagonal(sim_mat, 0)
            
            recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)

            minval = np.min(sim_mat) - 1.
            
            np.fill_diagonal(sim_mat, minval)
            
            num_test = sim_mat.shape[0]
            
            n = 8#split the test data in to n (here set n=8) parts to fit in a GPU with 11G memory; otherwise, multiple GPUs are needed so that the whole test dataset can fit in; this splitting doesnot change the results.
            h_n = int(num_test/n)
            
            r1 =  float(h_n)/(0.0+num_test)
            r2 =  1.0 - r1*(n-1)
            
            mapr = 0.0
            rp = 0.0
            
            for i in range(n-1):
                mapr += MAP_R(sim_mat[h_n*i:h_n*(i+1)], labels, labels[h_n*i:h_n*(i+1)])*r1
                rp += R_Precision(sim_mat[h_n*i:h_n*(i+1)], labels, labels[h_n*i:h_n*(i+1)])*r1
            
            mapr += MAP_R(sim_mat[(n-1)*h_n:], labels, labels[(n-1)*h_n:])*r2
            rp += R_Precision(sim_mat[(n-1)*h_n:], labels, labels[(n-1)*h_n:])*r2
        
        model.train()
        at.train()
        head.train()

        return recall_ks, mapr, rp


    def test_dim512(model_list, at_list, head_list, conf):

        for model in model_list:
            model.eval()
        for at in at_list:
            at.eval()   
        for head in head_list:
            head.eval()
       
        if conf.use_dataset=='SOP':

            if conf.use_dataset == 'SOP':
                dataset = MSBaseDataSet(conf, conf.datapth+'sop_test.txt', transform=conf.transform_dict['center-crop'], mode='RGB')

            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=False, pin_memory=True, drop_last=False)

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
                    
            feas = feas.cpu().numpy()
            
            sim_mat = np.matmul( feas, np.transpose(feas) )
 
            np.fill_diagonal(sim_mat, 0)
            
            recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)

            minval = np.min(sim_mat) - 1.
            
            np.fill_diagonal(sim_mat, minval)
            
            num_test = sim_mat.shape[0]
            
            n = 8#split the test data samples in to n (here set n=8) parts to fit in a GPU with 11G memory; otherwise, multiple GPUs are needed so that the whole test dataset can fit in; this splitting doesnot change the results.
            h_n = int(num_test/n)
            
            r1 =  float(h_n)/(0.0+num_test)
            r2 =  1.0 - r1*(n-1)
            
            mapr = 0.0
            rp = 0.0
            
            for i in range(n-1):
                mapr += MAP_R(sim_mat[h_n*i:h_n*(i+1)], labels, labels[h_n*i:h_n*(i+1)])*r1
                rp += R_Precision(sim_mat[h_n*i:h_n*(i+1)], labels, labels[h_n*i:h_n*(i+1)])*r1
            
            mapr += MAP_R(sim_mat[(n-1)*h_n:], labels, labels[(n-1)*h_n:])*r2
            rp += R_Precision(sim_mat[(n-1)*h_n:], labels, labels[(n-1)*h_n:])*r2
        
        model.train()
        at.train()
        head.train()
        
        return recall_ks, mapr, rp

    #start testing
    conf = get_config()
    if conf.test_pretrained_model!=1:
        
        model1 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_epoch{epoch}_{epoch_milestones}_batch{bs}_lr{lr}_ist{ist}_regcof{regcof}_atcof{atcof}_wcof{w_cof}.pth'.format(
                cross_id=1, seed=conf.seed, epoch=conf.epochs, epoch_milestones=conf.epoch_milestones, bs=conf.batch_size, lr=conf.lr, ist=conf.instances,
                regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof))               
        model1.load_state_dict(torch.load(model_path)['model'])
        at1 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at1.load_state_dict(torch.load(model_path)['at'])
        head1 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head1.load_state_dict(torch.load(model_path)['head'])
    
        model2 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_epoch{epoch}_{epoch_milestones}_batch{bs}_lr{lr}_ist{ist}_regcof{regcof}_atcof{atcof}_wcof{w_cof}.pth'.format(
                cross_id=2, seed=conf.seed+2, epoch=conf.epochs, epoch_milestones=conf.epoch_milestones, bs=conf.batch_size, lr=conf.lr, ist=conf.instances, 
                regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof))
        model2.load_state_dict(torch.load(model_path)['model'])
        at2 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at2.load_state_dict(torch.load(model_path)['at'])
        head2 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head2.load_state_dict(torch.load(model_path)['head']) 
    
        model3 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_epoch{epoch}_{epoch_milestones}_batch{bs}_lr{lr}_ist{ist}_regcof{regcof}_atcof{atcof}_wcof{w_cof}.pth'.format(
                cross_id=3, seed=conf.seed+4, epoch=conf.epochs, epoch_milestones=conf.epoch_milestones, bs=conf.batch_size, lr=conf.lr, ist=conf.instances,
                regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof))
        model3.load_state_dict(torch.load(model_path)['model'])
        at3 = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        at3.load_state_dict(torch.load(model_path)['at'])
        head3 = torch.nn.DataParallel( dim_reduce() ).cuda()
        head3.load_state_dict(torch.load(model_path)['head'])

        model4 = torch.nn.DataParallel( BNInception() ).cuda()
        model_path = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_epoch{epoch}_{epoch_milestones}_batch{bs}_lr{lr}_ist{ist}_regcof{regcof}_atcof{atcof}_wcof{w_cof}.pth'.format(
                cross_id=4, seed=conf.seed+6, epoch=conf.epochs, epoch_milestones=conf.epoch_milestones, bs=conf.batch_size, lr=conf.lr, ist=conf.instances,
                regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof))
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
     
    
