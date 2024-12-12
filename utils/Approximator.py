import torch
import time
import random
import numpy as np
import joblib
from utils.options import args_parser
from models.load_datasets import load_dataset
from models.load_models import load_model
from torchvision.models import resnet18
import joblib
import os
from utils.power_iteration import spectral_radius

args = args_parser()


def getapproximator(args,img_size,Dataset2recollect):
###############################################################################
#                               SETUP                                         #
###############################################################################
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
    print(args.num_classes)
    net = load_model(img_size).to(args.device)
    net.eval()    

    # load file
    path1 = "./Checkpoint/model_{}_checkpoints". format(args.model)
    file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat".format(
        args.dataset,args.epochs,args.lr,args.lr_decay,args.clip,args.seed)
    file_path = os.path.join(path1, file_name)
    info = joblib.load(file_path)  
    dataset = Dataset2recollect
    computed_rho = False
    # net setup
    lr=args.lr
    loss_func = torch.nn.CrossEntropyLoss()

###############################################################################
#                               Unlearning                                    #
###############################################################################
    # approximator
    approximator = {i: [torch.zeros_like(param) for param in net.parameters()] for i in range(len(dataset))}
    for step in range(len(info)):
        model_t, batch_idx = info[step]["model_list"],info[step]["batch_idx_list"]
        if args.model==resnet18:
            net.train()
        else:  net.eval()
        net.load_state_dict(model_t)
        batch_images_t, batch_labels_t = [], []
        for i in batch_idx:
            image_i, label_i, index_i = dataset[i]
            image_i ,label_i= image_i.unsqueeze(0).to(args.device), torch.tensor([label_i]).to(args.device)
            batch_images_t.append(image_i)
            batch_labels_t.append(label_i)
            log_probs = net(image_i)
            loss_i = loss_func(log_probs , label_i)
            net.zero_grad()
            for param in net.parameters():
                loss_i += 0.5 * args.regularization * (param * param).sum()
            loss_i.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.clip, norm_type=2)
            for j, param in enumerate(net.parameters()):
                approximator[i][j] += (param.grad.data * lr*(args.lr_decay**(step))) / len(batch_idx)
        log_probs=0
        loss_batch =0 
        grad_norm = 0
        batch_images_t = torch.cat(batch_images_t, dim=0)
        batch_labels_t = torch.cat(batch_labels_t, dim=0)
        log_probs = net(batch_images_t)
        loss_batch = loss_func(log_probs, batch_labels_t) 
        print("Recollecting Model  {:3d}, Training Loss: {:.2f}".format(step,loss_batch))
        for param in net.parameters():
            loss_batch += 0.5 * args.regularization * (param * param).sum()
        grad_params = torch.autograd.grad(loss_batch, net.parameters(), create_graph=True, retain_graph=True)
        grad_norm = torch.norm(torch.cat([grad.view(-1) for grad in grad_params]))
        if grad_norm > args.clip:
            scaling_factor = args.clip / grad_norm
            grad_params = [grad * scaling_factor for grad in grad_params]
        if not computed_rho:
            rho = spectral_radius(args, loss_batch=loss_batch, net=net,t=step)
            computed_rho = True  
        torch.cuda.synchronize()
        t_start = time.time()
        for i in range(len(dataset)): 
            net.zero_grad()
            HVP_i=torch.autograd.grad(grad_params, net.parameters(), approximator[i],retain_graph=True)
            for j, param in enumerate(net.parameters()):
                approximator[i][j]=approximator[i][j] - (lr* (args.lr_decay**(step)) * HVP_i[j].detach())
            del HVP_i # save memory
        del loss_batch,grad_params
        torch.cuda.synchronize()
        t_end = time.time()
        print("Computaion Time Elapsed:  {:.6f}s \n".format(t_end - t_start))
    del info,dataset


    return approximator, rho
    

