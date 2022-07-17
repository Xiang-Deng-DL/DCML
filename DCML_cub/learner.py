import myutils
import os
import torch

import numpy as np

from models.bninception import BNInception
from models.at_net import AT
from models.reduction import dim_reduce
from models.cos_net import CosNet

from torch.utils.data import DataLoader
from torch import optim

from evaluation import pairwise_similarity, Recall_at_ks, MAP_R, R_Precision
from data_engine import MSBaseDataSet, RandomIdSampler
import torch.nn as nn
import copy
from regularizer import RegLoss
from functions import weight_learner, inv_regv1, at_reg
import torch.nn.functional as F


class metric_learner(object):
    def __init__(self, conf, inference=False):
        
        self.model = torch.nn.DataParallel( BNInception() ).cuda()
        
        self.at = torch.nn.DataParallel( AT(at_num = conf.at_num) ).cuda()
        
        self.head = torch.nn.DataParallel( dim_reduce() ).cuda()
        
        self.proxy = torch.nn.DataParallel( CosNet(scale = conf.scale, dim_in = conf.dim, dim_out = conf.train_class) ).cuda() 
        
        self.erm_at = torch.nn.DataParallel( AT(at_num = 1) ).cuda()
        
        self.erm_proxy = torch.nn.DataParallel( CosNet(scale = conf.scale, dim_in = conf.dim, dim_out = conf.train_class) ).cuda() 
        
        
        self.ce =  nn.CrossEntropyLoss(reduce = False).cuda()
        self.erm_ce = nn.CrossEntropyLoss().cuda()
        self.reg = RegLoss(conf.instances).cuda()
        

        if not inference:

            if conf.use_dataset == 'CUB':
                train_path = conf.datapth+'cub_train'+str(conf.cross_valid)+'.txt'
                
                print(train_path)
  
                self.dataset = MSBaseDataSet(conf, train_path, transform=conf.transform_dict['rand-crop'], mode='RGB')
                
                valid_path = conf.datapth+'cub_valid'+str(conf.cross_valid)+'.txt'
                
                self.valid_dataset = MSBaseDataSet(conf, valid_path, transform=conf.transform_dict['center-crop'], mode='RGB')
           

            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=False, 
                sampler=RandomIdSampler(conf, self.dataset.label_index_dict), drop_last=False, pin_memory=True,)    
                    
            self.wl_dataloader = DataLoader(self.dataset, batch_size=conf.max_batch_size, num_workers=conf.num_workers, shuffle=False,
                                            sampler=RandomIdSampler(conf, self.dataset.label_index_dict), drop_last=False, pin_memory=True,)
            
            self.class_num = self.dataset.num_cls
            self.img_num = self.dataset.num_train
            
            self.epoch = 0

            backbone_bn_para, backbone_wo_bn_para = [
                [p for k, p in self.model.named_parameters() if
                 ('bn' in k) == is_bn] for is_bn in [True, False]]

            head_bn_para, head_wo_bn_para = [
                [p for k, p in self.head.named_parameters() if
                 ('bn' in k) == is_bn] for is_bn in [True, False]]
            
            self.optimizer = optim.RMSprop([
                    {'params': backbone_bn_para if conf.freeze_bn==False else [], 'lr': conf.lr_bone, 'momentum': 0.9},
                    {'params': backbone_wo_bn_para, 'weight_decay': conf.weight_decay, 'lr': conf.lr_bone, 'momentum': 0.9},
                    {'params': head_bn_para, 'lr': conf.lr_bone, 'momentum': 0.9},
                    {'params': head_wo_bn_para, 'weight_decay': conf.weight_decay, 'lr': conf.lr_bone, 'momentum': 0.9},    
                    {'params': self.at.parameters(), 'weight_decay': conf.weight_decay, 'lr': conf.lr_at, 'momentum': 0.9},
                    {'params': self.erm_at.parameters(), 'weight_decay': conf.weight_decay, 'lr': conf.lr_at, 'momentum': 0.9},                    
                    {'params': self.proxy.parameters(), 'weight_decay': conf.weight_decay, 'lr': conf.lr, 'momentum': 0.9},
                    {'params': self.erm_proxy.parameters(), 'weight_decay': conf.weight_decay, 'lr': conf.lr, 'momentum': 0.9},
                    ])

            print(f'{self.optimizer}, optimizers generated')

            if conf.use_dataset=='CUB':
                self.board_loss_every = 1  
                self.evaluate_every = 1
    

    def train(self, conf):
        
        weight_para = torch.randn((self.img_num, conf.at_num), requires_grad=True, device="cuda")# [data_num, evn_num]
        
        weight_para = weight_learner(self.model, self.at, self.head, self.proxy, self.ce, self.wl_dataloader, weight_para, conf)
        
        weight_para = weight_para.clone().detach()
        
        self.model.train()
        self.at.train()
        self.erm_at.train()
        self.head.train()
        self.proxy.train()
        self.erm_proxy.train()

        self.train_with_fixed_bn(conf)        
        
        myutils.timer.since_last_check('start train')
        data_time = myutils.AverageMeter(20)
        loss_time = myutils.AverageMeter(20)
        loss_meter = myutils.AverageMeter(20)

        self.epoch = conf.start_epoch
        
        if self.epoch == 0 and conf.start_eval:
            
            recall_ks, mapr, rp = self.test(conf)

            print(f'test on {conf.use_dataset}: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr} ')

            recall_ks, mapr, rp = self.validate(conf)

            print(f'val on {conf.use_dataset}: P@1 is {recall_ks[0]}, rp_val is {rp}, mapr_val is {mapr}')

            self.train_with_fixed_bn(conf)

        best_testrecall = 0
        bestval_recall = 0
        
        while self.epoch <= conf.epochs:
            
            loader_enum = enumerate(self.loader)
            
            if self.epoch == conf.epoch_milestones[0]:
                self.schedule_lr(conf)
            if self.epoch == conf.epoch_milestones[1]:
                self.schedule_lr(conf)
            if self.epoch == conf.epoch_milestones[2]:
                self.schedule_lr(conf)
            
            while True:
                if self.epoch >= conf.epochs:
                    break
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    print(f'one epoch finish {e} {ind_data}')
                    break
                
                data_time.update(myutils.timer.since_last_check(verbose=False))
                
                loss_ce = []
                loss_proxy = []
                loss_regs = []

                imgs = data['image'].to(conf.device)
                labels = data['label'].to(conf.device)
                index = data['index']

                self.optimizer.zero_grad()
                
                bbout = self.model(imgs)
                
                env_weights = F.softmax(weight_para[index], dim=-1)

                for i in range(conf.at_num):

                    batch_w = env_weights[:,i]
                    
                    fea = self.at(bbout, train=1, i = i)#32*1024*7*7
                    
                    fea = self.head(fea)
                
                    cos_sim = self.proxy(fea, labels)
                    
                    loss_ce.append( torch.sum(self.ce(cos_sim, labels)*batch_w )/batch_w.sum() )

                    inv = inv_regv1(cos_sim, labels, batch_w, self.ce, conf)
                    loss_proxy.append( inv  )
                    
                    reg_loss = self.reg(fea, labels, batch_w)
                    loss_regs.append(reg_loss)
                 
                
                loss_envce = sum(loss_ce)/len(loss_ce)

                loss_inv = sum(loss_proxy)/len(loss_proxy)
                
                loss_reg = sum(loss_regs)/len(loss_regs)
                
                loss = loss_envce + conf.w_cof*loss_inv + conf.at_cof*at_reg(self.at.module.atms, self.at.module.atls) + conf.reg_cof*loss_reg
                
                fea = self.erm_at(bbout, train=0)
                fea = self.head(fea)
                cos_sim = self.erm_proxy(fea, labels)
                loss_erm = self.erm_ce(cos_sim, labels)
                
                loss += loss_erm
                
                loss.backward()
                
                loss_meter.update(loss.item())
                
                self.optimizer.step()


            if (self.epoch+1)%conf.updata_env == 0:
                weight_para = weight_learner(self.model, self.at, self.head, self.proxy, self.ce, self.wl_dataloader, weight_para, conf)
                self.train_with_fixed_bn(conf)  
                
            if self.epoch % self.evaluate_every ==0 and self.epoch != 0:
                
                recall_ks, mapr, rp = self.test(conf)
                
                print(f'epoch {self.epoch} test on {conf.use_dataset}: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                
                if recall_ks[0] > best_testrecall:
                    
                    best_testrecall = recall_ks[0]
                    best_testrecalls = recall_ks
                    best_mapr = mapr
                    best_rp = rp
                    
                    print(f'best test recalls: P@1 is {recall_ks[0]}, RP is {rp}, MAP@Rr is {mapr}')
                                    
                
                recall_ks_val, mapr_val, rp_val = self.validate(conf)
                
                print(f'epoch {self.epoch} val on {conf.use_dataset}: P@1 is {recall_ks_val[0]}, rp_val is {rp_val}, mapr_val is {mapr_val}')
                              
                if recall_ks_val[0] > bestval_recall:
                    
                    bestval_recall = recall_ks_val[0]
                    
                    self.save_model(conf)#save the model when the validation acc is the best
                    
                    bestval_testrecalls = recall_ks
                    bestval_mapr = mapr
                    bestval_rp = rp
                    
                    print(f'bestval-select test recalls: P@1 is {recall_ks[0]}, RP is {rp}, MAP@R is {mapr}')
                
                self.train_with_fixed_bn(conf)

            if self.epoch % self.board_loss_every == 0 and self.epoch != 0:
                
                print(f'epoch {self.epoch}: ' +
                      f'loss: {loss_meter.avg:.3f} ' +
                      f'data time: {data_time.avg:.2f} ' +
                      f'loss time: {loss_time.avg:.2f} ' +
                      f'speed: {conf.batch_size/(data_time.avg+loss_time.avg):.2f} imgs/s '
                      )

                
            loss_time.update(myutils.timer.since_last_check(verbose=False))

            self.epoch += 1
            
        print(f'last best_test-selected test recall on {conf.use_dataset}: P@1 is {best_testrecalls[0]}, RP is {best_rp}, MAP@R is {best_mapr}')#only for print use
        
        print(f'last best_validation-selected test recall on {conf.use_dataset}: P@1 is {bestval_testrecalls[0]}, RP is {bestval_rp}, MAP@R is {bestval_mapr}')#we reprot the test acc selected by the validation set

    def save_model(self, conf):
        state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'at': self.at.state_dict(),
                'head': self.head.state_dict(),
                }
        save_file = os.path.join(conf.save_folder, 'valid_ckpt_crossid{cross_id}_seed{seed}_atnum{atnum}_regcof{regcof}_atcof{atcof}_wcof{w_cof}_maxw{maxw}max_at_cof{maxat_cof}.pth'.format(
                cross_id=conf.cross_valid, seed=conf.seed, atnum=conf.at_num, regcof=conf.reg_cof, atcof=conf.at_cof, w_cof=conf.w_cof, maxw=conf.maxw_cof, maxat_cof=conf.maxat_cof))
                
        torch.save(state, save_file)
    
    def train_with_fixed_bn(self, conf):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        if conf.freeze_bn:
            self.model.apply(fix_bn)
        else:
            pass

    def validate(self, conf):
        
        self.model.eval()
        self.at.eval()
        self.head.eval()        

        if conf.use_dataset == 'CUB':

            loader = DataLoader(self.valid_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
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

                    output1 = self.model(imgs)
                    output1 = self.at(output1, train=0)
                    output1 = self.head(output1)
                    
                    
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

        self.model.train()
        self.at.train()
        self.head.train()
        
        return recall_ks, mapr, rp

    def test(self, conf):
        
        self.model.eval()
        self.at.eval()
        self.head.eval()
       
        if conf.use_dataset=='CUB':

            if conf.use_dataset == 'CUB':
                dataset = MSBaseDataSet(conf, conf.datapth+'cub_test.txt', transform=conf.transform_dict['center-crop'], mode='RGB')

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

                    imgs = data['image']
                    label = data['label']
                    

                    output1 = self.model(imgs)
                    
                    output1 = self.at(output1, train = 0)
                    
                    output1 = self.head(output1)
                    
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

        self.model.train()
        self.at.train()
        self.head.train()

        return recall_ks, mapr, rp


    def load_bninception_pretrained(self, conf):
        model_dict = self.model.state_dict()
        my_dict = {'module.'+k: v for k, v in torch.load(conf.bninception_pretrained_model_path).items() if 'module.'+k in model_dict.keys()}
        print('################################## do not have pretrained:')
        for k in model_dict:
            if k not in my_dict.keys():
                print(k)
        print('##################################')
        model_dict.update(my_dict)
        self.model.load_state_dict(model_dict)

    def schedule_lr(self, conf):
        for params in self.optimizer.param_groups:
            params['lr'] = params['lr'] * conf.lr_gamma

 
