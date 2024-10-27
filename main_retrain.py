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
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from utils.options import args_parser
from models.Update_retrain import  train
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet,FashionCNN4
import torch.nn as nn
import torch.utils.data as data
from utils.subset import reduce_dataset_size,sample_dataset_size
from models.test import test_img,test_per_img
import utils.loading_data as dataset
from torch.utils.data import Subset
import shutil
from torchvision.models import resnet18

if __name__ == '__main__':

    ########### Setup
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
        args.bs = 64
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
        idx = np.random.permutation(data.shape[0])
        data, label = data[idx], label[idx]
        train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)
        args.test_train_rate = 1; args.epochs-=1; args.batch_size+=1
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
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn4' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = FashionCNN4().to(args.device)
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
    # copy weights
    w = net.state_dict()
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))

    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    info=[]
    step=0
    all_indices = list(range(len(dataset_train)))
    # indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    # indices_to_remain = all_indices - indices_to_unlearn
    # dataset_train = sample_dataset_size(dataset_train, random_seed=args.seed, indices_to_unlearn=indices_to_remain)
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    remaining_indices = list(set(all_indices) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)


    for iter in range(args.epochs):
        t_start = time.time()
        w, loss,lr,step = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,  learning_rate=lr, indices_to_unlearn=indices_to_unlearn)
        # w, loss,lr,step = train_1(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=remain_dataset,  learning_rate=lr)
        t_end = time.time()   

        # copy weight to net
        net.load_state_dict(w)
        # print accuracy
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
         
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
    print(" Retrained {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))
   
    # save model
    rootpath1 = './log/Retrain/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)
    torch.save(net.state_dict(),  rootpath1+ 'Retrain_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'
               .format(args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed))

    # save acc of test datset
    rootpath3 = './log/Retrain/acctest/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    accfile = open(rootpath3 + 'Retrain_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), "w")
    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath3 + 'Retrain_plot_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.png'.format(
         args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed))
    # save Loss
    rootpath4 = './log/Retrain/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'Retrain_lossfil_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), "w")
    for loss in loss_test:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath4 + 'Retrain_plot_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed))


    ##### Compute loss/acc
    # loss
    forget_acc_list, forget_loss_list = test_img(net, forget_dataset, args)
    rootpath = './log/Retrain/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
    args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    lossfile.write(str(forget_loss_list))
    lossfile.close()
    # acc 
    rootpath = './log/Retrain/accforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Retrain_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
   args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    accfile.write(str(forget_acc_list))
    accfile.close()

    ##### Compute loss/acc on remaining datase
    # loss of remaining data
    remain_acc_list, remain_loss_list = test_img(net, remain_dataset , args)
    rootpath = './log/Retrain/lossremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
    args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    lossfile.write(str(remain_loss_list))
    lossfile.close()
    # acc of forgetting data
    rootpath = './log/Retrain/accremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Retrain_accfile_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
    args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed), 'w')
    accfile.write(str(remain_acc_list))
    accfile.close()

 










