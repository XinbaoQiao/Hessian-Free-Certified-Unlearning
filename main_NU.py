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
