#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import joblib
import os
import copy
import random
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser
import torch
torch.set_printoptions(threshold=np.inf)


class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label, self.all_indices[item]


def train(step,args, net, dataset,learning_rate,info):

    # Ensure reproducibility of results, which may lead to a slight decrease in performance as it disables some optimizations.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    args = args
    loss_func = nn.CrossEntropyLoss()
    lr=learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    Dataset2recollect=DatasetSplit(dataset)
    dataloader = DataLoader(DatasetSplit(dataset), batch_size=args.batch_size, shuffle=True)
    
    loss=0
    for batch_idx, (images, labels, indices) in enumerate(dataloader):
        optimizer.zero_grad()
        net.eval()
        # save sample idx in batch
        info.append({"batch_idx_list": indices.tolist(), "model_list": copy.deepcopy(net).state_dict()})  
        images, labels = images.to(args.device), labels.to(args.device)
        net.zero_grad()
        log_probs = net(images)

        loss = loss_func(log_probs, labels)
        for param in net.parameters():
            loss += 0.5 * args.regularization * (param * param).sum()
        net.train()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.clip, norm_type=2)
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        del images, labels, log_probs
        print("     Step {:3d}     Batch {:3d}, Batch Size: {:3d}, Trainning Loss: {:.2f}".format(step,batch_idx,dataloader.batch_size,loss))
        step +=1

    return net.state_dict(), loss, lr, Dataset2recollect,step,info

