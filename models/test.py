#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
from utils.options import args_parser
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label, self.all_indices[item]

def test_img(net_g, datatest, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    net_g.eval()
    net_g.to(args.device)
    # testing
    total = 0
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(datatest), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target, indices) in enumerate(data_loader):
        with torch.no_grad():
            # if torch.cuda.is_available() and args.gpu != -1:
            #     data, target = data.to(args.device), target.to(args.device)
            # else:
            #     data, target = data.cpu(), target.cpu()
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # print(test_loss)
    return accuracy, test_loss

def test_per_img(net_g, dataset, args, indices_to_test):
    net_g.eval()
    # testing
    test_loss_list = []
    correct = 0
    sample_losses=0
    data_loader = DataLoader(DatasetSplit(dataset), batch_size=len(dataset), shuffle=False)
    for idx, (data, target,indices) in enumerate(data_loader):
        indices_to_test = [i for i in range(len(indices)) if indices[i] in indices_to_test]
        if len(indices_to_test) == 0:
            # Skip empty batch
            continue
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data[indices_to_test].cuda(args.device), target[indices_to_test].cuda(args.device)
        else:
            data, target = data[indices_to_test].cpu(), target[indices_to_test].cpu()
        if data.numel() == 0:
    # Skip empty data
            continue
        log_probs = net_g(data)
        # sum up batch loss
        sample_losses = F.cross_entropy(log_probs, target, reduction='none')
        test_loss_list.extend(sample_losses.tolist())
        # print(f"Index: {idx}, Loss: {sample_losses}")
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss_list 



