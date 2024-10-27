import torch
import time
import joblib
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet,FashionCNN4
import torch.nn as nn
import random
import numpy as np
import torch
import os
import torch
from torchvision.models import resnet18
from utils.power_iteration import spectral_radius


def getapproximator_resnet(args,img_size,Dataset2recollect,indices_to_unlearn):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.empty_cache()
    
    net = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn4' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = FashionCNN4().to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,2)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,10)
    elif args.model == 'resnet18' and args.dataset == 'svhn':
        net = resnet18(pretrained=True).to(args.device)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features,10)
    elif args.model == 'resnet18' and args.dataset == 'lfw':
        net = resnet18(pretrained=True).to(args.device)
        # for param in net.parameters():
        #     param.requires_grad = False
        fc_inputs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 29))
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

    net = net.to(args.device)
    # print(net)
    # net setup
    lr=args.lr
    computed_rho = False
    loss_func = torch.nn.CrossEntropyLoss()
    approximator = {i: [torch.zeros_like(param) for param in net.parameters()] for i in indices_to_unlearn}
    for iter in range(args.epochs):
        # load file
        path1 = "./Checkpoint/Resnet/model_{}_checkpoints". format(args.model)
        file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}_iter_{}.dat". format(
            args.dataset, args.epochs,args.lr,args.lr_decay,args.clip,args.seed,iter)
        file_path = os.path.join(path1, file_name)
        info = joblib.load(file_path)  
        dataset = Dataset2recollect
        
        # approximator
        for b in range(len(info)):
            net.zero_grad()
            model_t, batch_idx = info[b]["model_list"],info[b]["batch_idx_list"]
            if args.model==resnet18:
                net.train()
            else:  net.eval()
            net.load_state_dict(model_t)
            batch_images_t, batch_labels_t = [], []
            loss_batch = 0.0
            for i in batch_idx:
                image_i, label_i, index_i = dataset[i]
                image_i ,label_i= image_i.unsqueeze(0).to(args.device), torch.tensor([label_i]).to(args.device)
                batch_images_t.append(image_i)
                batch_labels_t.append(label_i)        
                if i in indices_to_unlearn:
                    log_probs = net(image_i)
                    loss_i = loss_func(log_probs , label_i)
                    net.zero_grad()
                    for param in net.parameters():
                        loss_i += 0.5 * args.regularization * (param * param).sum()
                    loss_i.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.clip, norm_type=2)
                    for j, param in enumerate(net.parameters()):
                        approximator[i][j] += (param.grad.data * lr*(args.lr_decay**(len(info)*iter+b))) / len(batch_idx)
            log_probs=0
            loss_batch =0 
            grad_norm = 0
            batch_images_t = torch.cat(batch_images_t, dim=0)
            batch_labels_t = torch.cat(batch_labels_t, dim=0)
            log_probs = net(batch_images_t)
            loss_batch = loss_func(log_probs, batch_labels_t)    
            print("Recollecting Model  {:3d}, Training Loss: {:.2f}".format(b,loss_batch))
            for param in net.parameters():
                loss_batch +=  0.5 *args.regularization * (param * param).sum()
            grad_params = torch.autograd.grad(loss_batch, net.parameters(), create_graph=True, retain_graph=True)
            grad_norm = torch.norm(torch.cat([grad.view(-1) for grad in grad_params]))
            if grad_norm > args.clip:
                scaling_factor = args.clip / grad_norm
                grad_params = [grad * scaling_factor for grad in grad_params]
            if not computed_rho:
                rho = spectral_radius(args, loss_batch=loss_batch, net=net)
                computed_rho = True     
            t_start = time.time()   
            for i in indices_to_unlearn: 
                net.zero_grad()
                HVP_i=torch.autograd.grad(grad_params,net.parameters(),approximator[i],retain_graph=True)
                for j, param in enumerate(net.parameters()):
                    approximator[i][j]=approximator[i][j] - (lr* (args.lr_decay**(len(info)*iter+b)) * HVP_i[j].detach())
            del HVP_i,loss_batch,grad_params
            t_end = time.time()
            print("Computaion For step {} Time Elapsed:  {:.8f}s \n".format(iter,t_end - t_start))    

    return approximator, rho
    

