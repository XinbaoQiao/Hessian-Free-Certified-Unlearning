
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet

def load_model(img_size):
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset != 'cifar':
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
    # elif args.model == 'resnet18' and args.dataset == 'celeba':
    #     net = resnet18(pretrained=True).to(args.device)
    #     fc_features = net.fc.in_features
    #     net.fc = nn.Linear(fc_features,2)
    # elif args.model == 'resnet18' and args.dataset == 'cifar':
    #     net = resnet18(pretrained=True).to(args.device)
    #     fc_features = net.fc.in_features
    #     net.fc = nn.Linear(fc_features,10)
    # elif args.model == 'resnet18' and args.dataset == 'svhn':
    #     net = resnet18(pretrained=True).to(args.device)
    #     fc_features = net.fc.in_features
    #     net.fc = nn.Linear(fc_features,10)
    # elif args.model == 'resnet18' and args.dataset == 'lfw':
    #     net = resnet18(pretrained=True).to(args.device)
    #     for param in net.parameters():
    #         param.requires_grad = False
    #     fc_inputs = net.fc.in_features
    #     net.fc = nn.Sequential(
    #         nn.Linear(fc_inputs, 256),
    #         nn.ReLU(),
    #         nn.Dropout(),
    #         nn.Linear(256, 29))
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
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total)," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))
    return net