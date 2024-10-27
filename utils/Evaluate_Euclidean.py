#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import os
from utils.options import args_parser


def Evaluate_Euclidean(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    rootpath = './log'
    # Load model parameters
    proposed_model_path = rootpath + '/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    # IJ_model_path = rootpath + '/IJ/Model/IJ_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    # NU_model_path = rootpath + '/NU/Model/NU_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    retrain_model_path = rootpath + '/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)

    proposed_model_state_dict = torch.load(proposed_model_path)
    # IJ_model_state_dict = torch.load(IJ_model_path)
    # NU_model_state_dict = torch.load(NU_model_path)
    retrain_model_state_dict = torch.load(retrain_model_path)

    # Move the model parameters to GPU
    proposed_model_weights = torch.cat([param.view(-1).to(args.device) for param in proposed_model_state_dict.values()])
    # IJ_model_weights = torch.cat([param.view(-1).to(args.device) for param in IJ_model_state_dict.values()])
    # NU_model_weights = torch.cat([param.view(-1).to(args.device) for param in NU_model_state_dict.values()])
    retrain_model_weights = torch.cat([param.view(-1).to(args.device) for param in retrain_model_state_dict.values()])

    # Calculate L2 norm difference
    l2_norm_diff_proposed = torch.norm(proposed_model_weights -  retrain_model_weights , 2)
    # l2_norm_diff_IJ = torch.norm(IJ_model_weights -  retrain_model_weights, 2)
    # l2_norm_diff_NU = torch.norm(NU_model_weights -  retrain_model_weights, 2)

    print("(Proposed) L2 norm difference:", l2_norm_diff_proposed.item())
    # print("(IJ) L2 norm difference:", l2_norm_diff_IJ.item())
    # print("(NU) L2 norm difference:", l2_norm_diff_NU.item())

    # save evaluation
    rootpath = './results/Euclidean/'
    filename = 'Evaluate_Euclidean_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.txt'.format(args.model, args.dataset, args.num_forget, args.epochs,args.lr,args.lr_decay, args.clip,args.seed)
    output_file_path = os.path.expanduser(os.path.join(rootpath, filename))
    os.makedirs(rootpath, exist_ok=True)
    # Open a file and write the results
    with open(output_file_path, 'w') as file:
        file.write("(Proposed) L2 norm difference: {}\n".format(l2_norm_diff_proposed.item()))
        # file.write("(IJ) L2 norm difference: {}\n".format(l2_norm_diff_IJ.item()))
        # file.write("(NU) L2 norm difference: {}\n".format(l2_norm_diff_NU.item()))

    print("Euclidean Distance saved to:", output_file_path)

def Evaluate_Euclidean_ResNet(args):
    rootpath = './log'
    # Load model parameters
    proposed_model_path = rootpath + '/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    retrain_model_path = rootpath + '/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    finetune_model_path = rootpath + '/Finetune/Model/Finetune_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    neggrad_model_path = rootpath + '/NegGrad/Model/NegGrad_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)


    proposed_model_state_dict = torch.load(proposed_model_path)
    retrain_model_state_dict= torch.load(retrain_model_path)
    finetune_model_state_dict = torch.load(finetune_model_path)
    neggrad_model_state_dict = torch.load(neggrad_model_path)


    # Move the model parameters to GPU
    proposed_model_weights = torch.cat([param.view(-1) for param in proposed_model_state_dict.values()])
    retrain_model_weights = torch.cat([param.view(-1) for param in retrain_model_state_dict.values()])
    finetune_model_weights = torch.cat([param.view(-1) for param in finetune_model_state_dict.values()])
    neggrad_model_weights = torch.cat([param.view(-1) for param in neggrad_model_state_dict.values()])


    proposed_model_weights = proposed_model_weights.to('cpu')
    retrain_model_weights =retrain_model_weights.to('cpu')
    finetune_model_weights = finetune_model_weights.to('cpu')
    neggrad_model_weights = neggrad_model_weights.to('cpu')


    # Calculate L2 norm difference
    l2_norm_diff_proposed = torch.norm(proposed_model_weights - retrain_model_weights, 2)
    l2_norm_diff_finetune = torch.norm(finetune_model_weights - retrain_model_weights, 2)
    l2_norm_diff_neggrad = torch.norm(neggrad_model_weights - retrain_model_weights, 2)


    print("(Proposed) L2 norm difference:", l2_norm_diff_proposed.item())
    print("(Finetune) L2 norm difference:", l2_norm_diff_finetune.item())
    print("(NegGrad) L2 norm difference:", l2_norm_diff_neggrad.item())


    # save evaluation
    rootpath = './results/Euclidean/'
    filename = 'Evaluate_Euclidean_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.txt'.format(args.model, args.dataset, args.num_forget, args.epochs,args.lr,args.lr_decay, args.clip,args.seed)
    output_file_path = os.path.expanduser(os.path.join(rootpath, filename))
    os.makedirs(rootpath, exist_ok=True)
    # Open a file and write the results
    with open(output_file_path, 'w') as file:
        file.write("(Proposed) L2 norm difference: {}\n".format(l2_norm_diff_proposed.item()))
        file.write("(Finetune) L2 norm difference: {}\n".format(l2_norm_diff_finetune.item()))
        file.write("(NegGrad) L2 norm difference: {}\n".format(l2_norm_diff_neggrad.item()))

    print("Euclidean Distance saved to:", output_file_path)

