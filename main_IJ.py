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
import torch
import os
from models.load_datasets import load_dataset
from models.load_models import load_model
from utils.NU import compute_hessian,compute_gradient_unlearn
from utils.options import args_parser
from models.Update_NU import  train
from models.test import test_img,test_per_img
from torch.utils.data import Subset
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

    path="./data"
    if not os.path.exists(path):
        os.makedirs(path)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset_train, dataset_test, args.num_classes = load_dataset()
    img_size = dataset_train[0][0].shape
    net = load_model(img_size)
    w = net.state_dict()


###############################################################################
#                               LEARNING                                      #
###############################################################################
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










