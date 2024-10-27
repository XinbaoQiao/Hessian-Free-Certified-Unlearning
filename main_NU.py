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
from utils.NU import compute_hessian,compute_gradient_unlearn
from utils.options import args_parser
from models.Update_NU import  train
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet
from utils.subset import reduce_dataset_size
from models.test import test_img,test_per_img
from torch.utils.data import Subset
import utils.loading_data as dataset
import shutil



if __name__ == '__main__':
    ########### Setup ###########
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    torch.cuda.empty_cache()
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
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
    elif args.dataset == 'celeba':
        args.num_classe = 2
        args.bs = 1024
        custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128, 128)),
                                            #transforms.Grayscale(),                                       
                                            #transforms.Lambda(lambda x: x/255.),
                                            transforms.ToTensor()])
        
        custom_transform = custom_transform

        dataset_train = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-train.csv',
                                      img_dir='./data/celeba/img_align_celeba/',
                                      transform=custom_transform)
        # valid_celeba= dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-valid.csv',
        #                               img_dir='./data/celeba/img_align_celeba/',
        #                               transform=custom_transform)
        dataset_test = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-test.csv',
                                     img_dir='./data/celeba/img_align_celeba/',
                                     transform=custom_transform)
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


    ########### Model training ###########
    # training
    acc_test = []
    loss_test = []
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
    
    ########### Compute unlearning statistics ###########
    all_indices = list(range(len(dataset_train)))
    print(len(dataset_train))
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    remaining_indices = list(set(all_indices) - set(indices_to_unlearn))
    remain_dataset = Subset(dataset_train, remaining_indices)
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    
    # print("Forget: ",indices_to_unlearn)
    # print("Remain: ",remaining_indices)
    save_path = './log/NU/statistics/average_hessian_{}_{}_{}_{}.pth'.format(args.model,args.dataset,args.epochs,args.seed)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        print("Calculate unlearning statistics")
        average_hessian_all=compute_hessian(args,copy.deepcopy(net).to(args.device),  Dataset2recollect, all_indices)
        torch.save(average_hessian_all, save_path)
    else:
        print("Load average_hessian_all, No Need to compute Hessian")
        average_hessian_all = torch.load(save_path)
        # average_hessian_all = average_hessian_all + 0.01 * torch.eye(average_hessian_all.size(0), device=average_hessian_all.device)

    ########### Unlearning ###########
    print("(NU) Begin unlearning")
    average_hessian_forget=compute_hessian(args,copy.deepcopy(net).to(args.device),  Dataset2recollect, indices_to_unlearn)
    average_hessian_all = average_hessian_all.to(args.device)
    average_hessian_forget= average_hessian_forget.to(args.device)
    average_hessian = (average_hessian_all *  len(all_indices) - average_hessian_forget*  len(indices_to_unlearn) )/ len(remaining_indices) 
    average_hessian = average_hessian.to(args.device)
    # Compute the inverse of the average Hessian
    unlearn_t_start = time.time()
    inv_average_hessian = torch.inverse(average_hessian)
    inv_average_hessian=(inv_average_hessian/ len(remain_dataset))
    inv_average_hessian=inv_average_hessian.to(args.device)
    gradient_unlearn=compute_gradient_unlearn(args,copy.deepcopy(net).to(args.device),  forget_dataset)
    Approximator_NU= inv_average_hessian.mv(gradient_unlearn).to(inv_average_hessian.device)
    model_params = net.state_dict()
    for j, param in enumerate(net.parameters()):
        param.data += Approximator_NU[j]
    unlearn_t_end = time.time()

    acc_t, loss_t = test_img(net, dataset_test, args)
    acc_test.append(acc_t.item())
    loss_test.append(loss_t)

    print("(NU) Unlearned {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, unlearn_t_end - unlearn_t_start))



    ########### Save ###########
    # save unlearned model
    rootpath1 = './log/NU/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)  
    torch.save(net.state_dict(),  rootpath1+ 'NU_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save approximator
    rootpath2 = './log/NU/Approximator/'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2) 
    torch.save(Approximator_NU, rootpath2+ 'NU_Approximator_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    ###  Testing data 
    # save acc of test data
    rootpath3 = './log/NU/acctest/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3) 
    accfile = open(rootpath3 + 'NU_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")
    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath3 + 'NU_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    # save loss of test data
    rootpath4 = './log/NU/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'NU_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")
    for loss in loss_test:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
    # plot loss curve of test data
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath4 + 'NU_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    ###  Forgetting data 
    _, test_loss_list = test_per_img(net, dataset_train, args,indices_to_test=indices_to_unlearn)
    # Compute loss of forgetting data
    rootpath = './log/NU/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)    
    lossfile = open(rootpath + 'NU_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    for loss in test_loss_list:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
