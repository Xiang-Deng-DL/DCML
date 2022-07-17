import argparse
import torch
from torchvision import transforms
import myutils

def get_config():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--use_dataset', type=str, default='CUB', choices=['CUB'])
    
    parser.add_argument('--seed', type=int, default=2021)
    # batch
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--instances', type=int, default=4)
    parser.add_argument('--max_batch_size', type=int, default=64)
    
    # optimization
    parser.add_argument('--lr', type=float, default=2.53e-3)
    parser.add_argument('--lr_at', type=float, default=1e-6)
    parser.add_argument('--lr_bone', type=float, default=1e-6)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
     
    parser.add_argument('--at_num', type=int, default=2)
    parser.add_argument('--updata_env', type=int, default=5)
    

    parser.add_argument('--reg_cof', type=float, default=1e-2)
    parser.add_argument('--at_cof', type=float, default=1e-5)
    parser.add_argument('--w_cof', type=float, default=1e-5)  


    parser.add_argument('--max_optimizer', type=str, default='adam')
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--wleaner_lr', type=float, default=5e-1)
    parser.add_argument('--maxw_cof', type=float, default=1.0)
    parser.add_argument('--maxat_cof', type=float, default=1.0)
    parser.add_argument('--random_init', type=int, default=1)    

 
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epoch_milestones', type=str, default=None)
    
    # model dataset
    parser.add_argument('--freeze_bn', type=int, default=1)
    # method
    parser.add_argument('--use_loss', type=str, default='cos', choices=['cos'])

    parser.add_argument('--margin', type=float, default=0.6182)
    parser.add_argument('--scale', type=float, default=100.0)
    parser.add_argument('--train_class', type=int, default=75)
    parser.add_argument('--dim', type=int, default=128)
   
    parser.add_argument('--save_folder', type=str, default='./save')
    
    parser.add_argument('--cross_valid', type=int)
    
    parser.add_argument('--datapth', type=str, default='/home/xiang/data/cub-200-2011_metric/CUB_200_2011/')

    parser.add_argument('--bninception_pretrained_model_path', type=str, default='/home/xiang/SphericalEmbedding-main/pretrained_models/bn_inception-52deb4733.pth')

    parser.add_argument('--test_pretrained_model', type=int, default=0, choices=[0, 1])#test the pretrained metric models, default false

    conf = parser.parse_args()
    
    myutils.mkdir_p(conf.save_folder, delete=False)

    conf.num_devs = 1

    if conf.use_dataset == 'CUB':
        conf.start_epoch = 0

    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf.start_eval = True

    conf.num_workers = 8

    conf.transform_dict = {}
    conf.use_simple_aug = False

    conf.transform_dict['rand-crop'] = \
        transforms.Compose([
            transforms.Resize(size=(256, 256)) if conf.use_simple_aug else transforms.Resize(size=256),
            transforms.RandomCrop((227, 227)) if conf.use_simple_aug else transforms.RandomResizedCrop(
                                                                              scale=[0.16, 1],
                                                                              size=227
                                                                          ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123 / 255.0, 117 / 255.0, 104 / 255.0],
                                 std=[1.0 / 255, 1.0 / 255, 1.0 / 255]),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]) #to BGR
        ])
    conf.transform_dict['center-crop'] = \
        transforms.Compose([
            transforms.Resize(size=(256, 256)) if conf.use_simple_aug else transforms.Resize(size=256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123 / 255.0, 117 / 255.0, 104 / 255.0],
                                 std=[1.0 / 255, 1.0 / 255, 1.0 / 255]),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]) #to BGR
        ])
    

    return conf
