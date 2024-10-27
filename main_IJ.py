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
import torch
import os
from utils.IJ import compute_hessian,compute_gradient_unlearn
from utils.options import args_parser
from models.Update_NU import  train
from models.Nets import MLP, CNNMnist, CNNCifar,LeNet,Logistic
from utils.subset import reduce_dataset_size
from models.test import test_img,test_per_img
from torch.utils.data import Subset
import utils.loading_data as dataset
import shutil




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
        train_dataset = datasets.SVHN(root='data/',
                                    split='train',
                                    transform=train_transform,
                                    download=True)

        extra_dataset = datasets.SVHN(root='data/',
                                    split='extra',
                                    transform=train_transform,
                                    download=True)
    else:
        exit('Error: unrecognized dataset')

    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    img_size = dataset_train[0][0].shape

    net = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
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
    total_dimensions = 0
    total = sum(p.numel() for p in net.parameters()) 
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))
    # copy weights
    w = net.state_dict()

    ########### Model training
    # training
    acc_test = []
    loss_test=[]
    lr = args.lr
    info=[]
    step=0
    for iter in range(args.epochs):
        t_start = time.time()
        w, loss,lr,step,Dataset2recollect = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,learning_rate=lr)
        t_end = time.time()   

        # copy weight to net
        net.load_state_dict(w)
        # print accuracy
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
         
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)




    ########### Compute unlearning statistics
    all_indices = list(range(len(dataset_train)))
    print(len(dataset_train))
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    remaining_indices = list(set(all_indices) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)
    
    # print("Forget: ",indices_to_unlearn)
    # print("Remain: ",remaining_indices)
    save_path = './log/IJ/statistics/inv_average_hessian_{}_{}_{}_{}.pth'.format(args.model,args.dataset,args.epochs,args.seed)
    save_path2 = './log/NU/statistics/average_hessian_{}_{}_{}_{}.pth'.format(args.model,args.dataset,args.epochs,args.seed)
    unlearn_t_start = time.time()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if not os.path.exists(save_path2):
        if not os.path.exists(save_path):
            print("(IJ) Calculate unlearning statistics")
            average_hessian =compute_hessian(args,copy.deepcopy(net).to(args.device),  Dataset2recollect, all_indices)
            # Compute the inverse of the average Hessian
            inv_average_hessian = torch.inverse(average_hessian)
            save_path = './log/IJ/statistics/inv_average_hessian_{}_{}_{}_{}_{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
            torch.save(inv_average_hessian, save_path)
        else:
            print("(IJ) Load inv_average_hessian, No need to compute Hessian")
            inv_average_hessian =torch.load(save_path).to(args.device)
    else:
        print("Load average_hessian_all, No need to compute Hessian")
        average_hessian= torch.load(save_path2)
        # average_hessian = average_hessian + 0.01 * torch.eye(average_hessian.size(0), device=average_hessian.device)
        print('average_hessian device',average_hessian.device)
        # inv_average_hessian = torch.inverse(average_hessian)
        

    ########### Unlearning
    print("(IJ) Begin unlearning")

    average_hessian = average_hessian.to(args.device)
    torch.cuda.synchronize()
    unlearn_t_start = time.time()
    inv_average_hessian = torch.inverse(average_hessian)
    inv_average_hessian = inv_average_hessian.to(args.device)
    print('inv_average_hessian1 device',inv_average_hessian.device)
    # inv_average_hessian = inv_average_hessian.to('cpu')
    model_params = net.state_dict()
    gradient_unlearn = compute_gradient_unlearn(args,copy.deepcopy(net).to(args.device),  forget_dataset)
    # gradient_unlearn = gradient_unlearn.to('cpu')
    # print('inv_average_hessian device',inv_average_hessian.device)
    # print('unlearn gradient device',gradient_unlearn.device)
    Approximator_IJ = ((inv_average_hessian.mv(gradient_unlearn) )/ len(dataset_train)).to(inv_average_hessian.device)
    # Approximator_IJ.to(args.device)
    for j, param in enumerate(net.parameters()):
        param.data += Approximator_IJ[j]
    torch.cuda.synchronize()
    unlearn_t_end = time.time()
    acc_t, loss_t = test_img(net, dataset_test, args)
    acc_test.append(acc_t.item())
    loss_test.append(loss_t)
    print("(IJ) Unlearned {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.8f}s \n".format(iter, acc_t, unlearn_t_end - unlearn_t_start))


    ########### Save
    # save unlearned model
    rootpath1 = './log/IJ/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)  
    torch.save(net.state_dict(),  rootpath1+ 'IJ_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save approximator
    rootpath2 = './log/IJ/Approximator/'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2) 
    torch.save(Approximator_IJ, rootpath2+ 'IJ_Approximator_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))


    ###  Testing data 
    # save acc of testing data
    rootpath3 = './log/IJ/ACC/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3) 
    accfile = open(rootpath3 + 'IJ_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    # plot acc curve of testing data
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath3 + 'IJ_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    # save Loss of testing data
    rootpath4 = './log/IJ/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'IJ_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")
    for loss in loss_test:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
    # plot loss curve of testing data
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath4 + 'IJ_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    
    ###  Forgetting data 
    _, test_loss_list = test_per_img(net, dataset_train, args,indices_to_test=indices_to_unlearn)
    # Compute loss of forgetting data
    rootpath = './log/IJ/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)    
    lossfile = open(rootpath + 'IJ_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    for loss in test_loss_list:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()










