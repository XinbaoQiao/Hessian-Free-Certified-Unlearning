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
from models.load_datasets import load_dataset
from models.load_models import load_model
import torch
import os
from utils.options import args_parser
from models.Update_retrain import  train, train_1
from torch.utils.data import Subset
from models.test import test_img
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


###############################################################################
#                               LEARNING                                      #
###############################################################################
    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    info=[]
    step=0
    all_indices = list(range(len(dataset_train)))
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

 










