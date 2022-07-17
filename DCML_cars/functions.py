import torch
from torch import autograd
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


def inv_regv1(cos_sim, labels, batch_w, loss_function, conf):

    w = torch.ones((1, cos_sim.size(-1))).cuda().requires_grad_()

    loss = torch.sum(loss_function(cos_sim*w, labels)*batch_w)/batch_w.sum()

    grad = autograd.grad(loss, [w], create_graph=True)[0]
    
    return torch.sum(grad**2)


def max_invat(at_map, head, proxy, labels, batch_w, loss_function, conf):
    
    bs, c_num, w_map, h_map = at_map.shape
    
    at_map1 = torch.ones( ( 1, c_num, 1, 1 ) ).cuda().requires_grad_()
    
    at_map2 = torch.ones( ( 1, 1, w_map, w_map ) ).cuda().requires_grad_()
    
    at_map = at_map*at_map2*at_map1
    
    cos_sim = proxy( head(at_map), labels )

    loss = torch.sum(loss_function(cos_sim, labels)*batch_w)/batch_w.sum()

    grad1 = autograd.grad(loss, [at_map2], create_graph=True)[0]
    
    grad2 = autograd.grad(loss, [at_map1], create_graph=True)[0]
    
    return torch.sum(grad1**2)+torch.sum(grad2**2)


def at_reg(atms, atls):
    
    assert len(atms) == len(atls)
    
    at_num = len(atms)
    
    w1_mean = torch.stack([atms[i].fc1.weight for i in range(at_num)], 0).mean(0)
    
    w2_mean = torch.stack([atms[i].fc2.weight for i in range(at_num)], 0).mean(0)
    
    w3_mean = torch.stack([atls[i].fc1.weight for i in range( at_num)], 0).mean(0)
    
    w4_mean = torch.stack([atls[i].fc2.weight for i in range( at_num)], 0).mean(0)
    
    w_vars = []
    
    for i in range(at_num):
        w_vars.append( (torch.norm(atms[i].fc1.weight - w1_mean, p=2) / torch.norm(atms[i].fc1.weight, p=1) ) ** 2 )
        w_vars.append( (torch.norm(atms[i].fc2.weight - w2_mean, p=2) / torch.norm(atms[i].fc2.weight, p=1) ) ** 2 )
        w_vars.append( (torch.norm(atls[i].fc1.weight - w3_mean, p=2) / torch.norm(atls[i].fc1.weight, p=1) ) ** 2 )
        w_vars.append( (torch.norm(atls[i].fc2.weight - w4_mean, p=2) / torch.norm(atls[i].fc2.weight, p=1) ) ** 2 )
    
    loss = sum(w_vars) / len(w_vars)
    
    return loss




def weight_learner(model, at, head, proxy, ce, data_loader, weight_para, conf):
    
    img_num = weight_para.shape[0]
    
    if conf.random_init:
        weight_para = torch.randn((img_num, conf.at_num), requires_grad=True, device="cuda")
    else:
        weight_para = weight_para.requires_grad_()
    
    cnt = 0
    num_env = weight_para.size(1)

    # optimizer and schedule
    if conf.max_optimizer == "SGD":
        wl_optimizer = torch.optim.SGD([weight_para], lr= conf.wleaner_lr, momentum=0.9, weight_decay=0.) # 0.1
        scheduler = MultiStepLR(wl_optimizer, [5, 10], gamma=0.2, last_epoch=-1)
    
    if conf.max_optimizer == "adam":
        wl_optimizer = torch.optim.Adam([weight_para], lr = conf.wleaner_lr, weight_decay=0.)
        scheduler = MultiStepLR(wl_optimizer, [10, 15], gamma=0.2, last_epoch=-1)

    model.eval()
    at.eval()
    head.eval()
    proxy.eval()
    
    for param in head.parameters():
        param.requires_grad = False

    for param in proxy.parameters():
        param.requires_grad = False

    for epoch in range(1, conf.max_epoch+1):
        
        loss_list, loss_invs_list = [], []

        loader_enum = enumerate(data_loader)
        
        while True:
            
            try: 
                
                data_index, data = loader_enum.__next__()
            
            except StopIteration as e:
                
                #print(f'one epoch finish {e} {data_index}')                
                break       

            img = data['image'].cuda()
            labels = data['label'].cuda()
            idx = data['index']
            
            with torch.no_grad():
                
                bbout = model(img)
                
            loss_inv_list, loss_invat_list = [], []

            weights = F.softmax(weight_para[idx], dim=-1)
            
            for i in range(num_env):
                
                #print('labels', labels)
                with torch.no_grad():
                    
                    fea1 = at(bbout, train=1, i=i)
                    
                    fea = head(fea1)
                    
                    cos_sim = proxy(fea, labels)
                
                batch_w = weights[:, i]
                        
                inv = inv_regv1(cos_sim, labels, batch_w, ce, conf)
                
                loss_inv_list.append(inv)
                
                invat = max_invat(fea1, head, proxy, labels, batch_w, ce, conf)
                
                loss_invat_list.append(invat)
            
            
            loss_inv = torch.stack(loss_inv_list).mean()
            loss_invat = torch.stack(loss_invat_list).mean()
            
            loss_final = - (conf.maxw_cof*loss_inv + conf.maxat_cof*loss_invat)


            wl_optimizer.zero_grad()
            loss_final.backward()
            wl_optimizer.step()

            loss_list.append(loss_final.item())
            loss_invs_list.append(-loss_inv.item())

        scheduler.step()
        
        avg_loss = sum(loss_list)/len(loss_list)

        if epoch == 1:
            best_loss = avg_loss
            weight_para_best = weight_para.clone().detach()
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            weight_para_best = weight_para.clone().detach()

        for param in head.parameters():
            param.requires_grad = True
            
        for param in proxy.parameters():
            param.requires_grad = True
            
        model.train()
        at.train()
        head.train()
        proxy.train()
        
        return weight_para_best
