#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import copy
import numpy as np
from models.load_datasets import load_dataset
from models.load_models import load_model
import torch
import os
from utils.Approximator import getapproximator
from utils.Approximator_resnet import  getapproximator_resnet
from utils.options import args_parser
from utils.perturbation import NoisedNetReturn
from models.Update import  train
from torch.utils.data import Subset
from models.test import test_img
import shutil
import joblib

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
    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    step=0
    info=[]
    for iter in range(args.epochs):
        torch.cuda.synchronize()
        t_start = time.time()
        w, loss,lr,Dataset2recollect,step,info = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,learning_rate=lr,info=info)
        torch.cuda.synchronize()
        t_end = time.time()   
        net.load_state_dict(w)
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.7f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)

    rootpath = './log/Original/Model/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)   
    torch.save(net.state_dict(),  rootpath+ 'Original_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'
               .format(args.model,args.dataset, args.epochs,args.lr,args.lr_decay,args.clip,args.seed))
    
    all_indices_train = list(range(len(dataset_train)))
    indices_to_unlearn = random.sample(all_indices_train, k=args.num_forget)
    remaining_indices = list(set(all_indices_train) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)

###############################################################################
#                              PRECOMPUTATION UNLEARNING                      #
###############################################################################
    # load file
    path1 = "./Checkpoint/model_{}_checkpoints". format(args.model)
    file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat". format(
        args.dataset,args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    file_path = os.path.join(path1, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    info = joblib.dump(info,file_path); rho=0  
    save_path = './log/Proposed/statistics/Approximators_all_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model,args.dataset,args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    if args.model in ['resnet18']:
        Approximators, rho =getapproximator_resnet(args,img_size,Dataset2recollect=Dataset2recollect,indices_to_unlearn=indices_to_unlearn)
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
    torch.cuda.synchronize()
    unlearn_t_start = time.time()
    for idx in indices_to_unlearn:
        for j in range(len(Approximator_proposed)):
            Approximator_proposed[j] += Approximators[idx][j]
## ------------ Online Unlearning ------------
        for j, param in enumerate(net.parameters()):
            param.data += Approximator_proposed[j]
            Approximator_proposed[j].zero_() 
        if args.application ==True:  
            w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=rho, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
            net.load_state_dict(w)
## ------------ Online Unlearning ------------
    # del Approximators
    # torch.cuda.synchronize()
    # unlearn_t_start = time.time()
# ------------ Batch Unlearning ------------
    # for j, param in enumerate(net.parameters()):
    #     param.data += Approximator_proposed[j]
    # if args.application ==True:
    #     w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=rho, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset,m=args.num_forget)
    #     net.load_state_dict(w)
# ------------ Batch Unlearning ------------
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



