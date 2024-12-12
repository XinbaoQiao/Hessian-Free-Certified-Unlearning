#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import cv2
import torch
import os
from utils.Approximator import getapproximator
from utils.Approximator_resnet import getapproximator_resnet
from utils.options import args_parser
from utils.subset import reduce_dataset_size
from utils.perturbation import NoisedNetReturn
import utils.loading_data as dataset
from models.Update import  train
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet
import torch.nn as nn
# resnet18
from torch.utils.data import Subset
import torch.utils.data as data
from models.test import test_img, test_per_img
import shutil
import joblib
from torchvision.models import resnet18

if __name__ == '__main__':

###############################################################################
#                               SETUP                                         #
###############################################################################
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    path2="./data"
    if not os.path.exists(path2):
        os.makedirs(path2)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])     
        # transform = transforms.Compose([transforms.Resize((224, 224)),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
        #                        )
                               
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    elif args.dataset == 'celeba':
        args.num_classe = 2
        # args.bs = 64
        custom_transform =transforms.Compose([
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        dataset_train = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-train.csv',
                                      img_dir='./data/celeba/img_align_celeba/',
                                      transform=custom_transform)
        # valid_celeba= dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-valid.csv',
        #                               img_dir='./data/celeba/img_align_celeba/',
        #                               transform=custom_transform)
        dataset_test = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-test.csv',
                                     img_dir='./data/celeba/img_align_celeba/',
                                     transform=custom_transform)
    elif args.dataset == 'svhn':
        num_classes = 10
        train_transform = transforms.Compose([])
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        dataset_train = datasets.SVHN(root='./data/svhn',
                                    split='train',
                                    transform=train_transform,
                                    download=True)

        dataset_test = datasets.SVHN(root='./data/svhn',
                                    split='test',
                                    transform=train_transform,
                                    download=True)
    elif args.dataset == 'lfw':
        num_classes = 29
        path = './data/lfw'
        pathlist = map(lambda x: '/'.join([path, x]), os.listdir(path))
        namedict = {}
        data, label = [], []
        idx = 0
        for item in pathlist:
            dirlist = os.listdir(item)
            if not (30<= len(dirlist) <= 100):
                continue
            for picpath in dirlist:
                data.append(cv2.imread(item + '/' + picpath))
                label.append(idx)
            namedict[str(idx)] = item.split('/')[-1]
            idx += 1
        data, label = np.stack(data), np.array(label)
        args.test_train_rate = 1; args.epochs-=1; args.batch_size+=1
        idx = np.random.permutation(data.shape[0])
        data, label = data[idx], label[idx]
        train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)
        dataset_train = dataset.LFWDataSet(train_X, train_Y)
        dataset_test= dataset.LFWDataSet(test_X, test_Y)
        args.num_dataset = len(dataset_train)

    else:
        exit('Error: unrecognized dataset')

    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    img_size = dataset_train[0][0].shape
    print('Train dataset:   ',len(dataset_train))
    print('Test dataset:   ',len(dataset_test))
        
    net = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset != 'cifar':
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,2)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,10)
    elif args.model == 'resnet18' and args.dataset == 'svhn':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,10)
    elif args.model == 'resnet18' and args.dataset == 'lfw':
        net = resnet18(pretrained=True).to(args.device)
        # for param in net.parameters():
        #     param.requires_grad = False
        fc_inputs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 29))
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'logistic':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = Logistic(dim_in=len_in,  dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net)
    net.train()
    w = net.state_dict()
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))


    ###################### Learning ######################
    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    step=0

    for iter in range(args.epochs):
        info=[]
        t_start = time.time()
        w, loss,lr,Dataset2recollect,step,info = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,learning_rate=lr,info=info)
        t_end = time.time()   
        # copy weight to net
        net.load_state_dict(w)
        # print accuracy
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.8f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
        path1 = "./Checkpoint/Resnet/model_{}_checkpoints". format(args.model)
        file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}_iter_{}.dat".format(
            args.dataset,args.epochs,args.lr,args.lr_decay,args.clip,args.seed,iter)
        file_path = os.path.join(path1, file_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        info = joblib.dump(info,file_path)
        del info 

    rootpath = './log/Original/Model/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)   
    torch.save(net.state_dict(),  rootpath+ 'Original_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'
               .format(args.model, args.dataset, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))  

    all_indices_train = list(range(len(dataset_train)))
    indices_to_unlearn = random.sample(all_indices_train, k=args.num_forget)
    remaining_indices = list(set(all_indices_train) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)


###############################################################################
#                              PRECOMPUTATION UNLEARNING                      #
###############################################################################
    save_path = './log/Proposed/statistics/Approximators_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model,args.dataset,args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    Approximator_proposed = {j: torch.zeros_like(param) for j, param in enumerate(net.parameters())}
    if args.model in ['resnet18']:
        for i in range(0, len(indices_to_unlearn), 50):
            indices_to_unlearn_i = indices_to_unlearn[i:i+50]
            Approximators,rho=getapproximator_resnet(args,img_size,Dataset2recollect=Dataset2recollect,indices_to_unlearn=indices_to_unlearn_i)
            del Dataset2recollect
    else:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if not os.path.exists(save_path):
            print("Calculate unlearning statistics")
            Approximators,rho=getapproximator(args,img_size,Dataset2recollect=Dataset2recollect)
            del Dataset2recollect
            torch.save({'Approximators': Approximators, 'rho': rho}, save_path)
        else:
            print("Load approximator")
            data = torch.load(save_path)
            Approximators = data['Approximators']
            rho = data['rho']

###############################################################################
#                               UNLEARNINTG                                   #
###############################################################################
    print("(Proposed) Begin unlearning")
    Approximator_proposed = {j: torch.zeros_like(param) for j, param in enumerate(net.parameters())}
    for idx in indices_to_unlearn:
        for j in range(len(Approximator_proposed)):
            Approximator_proposed[j] += Approximators[idx][j]
# ## ------------ Online Unlearning ------------
#         for j, param in enumerate(net.parameters()):
#             param.data += Approximator_proposed[j]
#             Approximator_proposed[j].zero_() 
#         if args.application ==True:  
#             w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=rho, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
#             net.load_state_dict(w)
# ## ------------ Online Unlearning ------------
    del Approximators
    torch.cuda.synchronize()
    unlearn_t_start = time.time()
## ------------ Batch Unlearning ------------
    for j, param in enumerate(net.parameters()):
        param.data += Approximator_proposed[j]
    if args.application ==True:
        w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=rho, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset,m=args.num_forget)
        net.load_state_dict(w)
## ------------ Batch Unlearning ------------
    torch.cuda.synchronize()
    unlearn_t_end = time.time()
    acc_t, loss_t = test_img(net, dataset_test, args)
    acc_test.append(acc_t.item())
    loss_test.append(loss_t)
    print("(Proposed) Unlearned {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.8f}s \n".format(iter, acc_t, unlearn_t_end - unlearn_t_start))


###############################################################################
#                               SAVE                                          #
###############################################################################
    # save unlearned model
    rootpath1 = './log/Proposed/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)   
    torch.save(net.state_dict(),  rootpath1+ 'Proposed_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'
               .format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))

    # save approximator
    rootpath2 = './log/Proposed/Approximator/'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2)    
    torch.save(Approximator_proposed,  rootpath2+ 'Proposed_Approximator_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
                        args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))

    ### Test Data
    # save acc of test sample
    rootpath3 = './log/Proposed/acctest/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    accfile = open(rootpath3 + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.
                   format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed), "w")
    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    # plot acc curve of test sample
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath3 + 'Proposed_plot_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.png'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))
    # save loss of test sample
    rootpath4 = './log/Proposed/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.
                   format( args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed), "w")
    for loss in loss_test:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
    # plot loss curve of test sample
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath4 + 'Proposed_plot_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.png'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))
    

    ### Forgetting data 
    forget_acc, forget_loss = test_img(net, forget_dataset, args)
    # acc of forgetting data
    rootpath = './log/Proposed/accforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
    args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    accfile.write(str(forget_acc))
    accfile.close()

    ###  Remaining data 
    remain_acc_list, remain_loss_list = test_img(net, remain_dataset , args)
    # acc of remain data
    rootpath = './log/Proposed/accremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
    args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    accfile.write(str(remain_acc_list))
    accfile.close()
