import torch
from torch.autograd.functional import hessian
import torch.nn as nn
from torch.autograd import grad, Variable
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')
from torch import nn
import torch
import time
from utils.options import args_parser
from torch.utils.data import DataLoader, Dataset
args = args_parser()



class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label, self.all_indices[item]




def compute_hessian(args, model, Dataset2recollect, all_indices):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    image_0, label_0, index_0 = Dataset2recollect[all_indices[0]]
    total_hessian = torch.zeros_like(calc_hessian(args,loss_func(model(image_0.unsqueeze(0).to(args.device)),torch.tensor([label_0]).to(args.device)), model.parameters()))   
    step = 0
    
    for i in all_indices:
        t_start = time.time() 
        image_i, label_i, index_i = Dataset2recollect[i]
        image_i, label_i = image_i.unsqueeze(0).to(args.device), torch.tensor([label_i]).to(args.device)
        log_probs = model(image_i)
        loss_i = loss_func(log_probs, label_i)
        hessian_i = calc_hessian(args, loss_i, model.parameters())
        total_hessian += hessian_i.detach()  
        del hessian_i
        step += 1
        model.zero_grad()
        t_end = time.time() 
        print("Calculating the Hessian of the {:3d}-th data point, Time Elapsed: {:.2f}s".format(step, t_end - t_start))

    # compute average hessian
    average_hessian = total_hessian / len(all_indices)

    if args.model != 'logistic':
        print("(IJ) Hessian matrix is singular, thus a damping factor of {:.5f} ".format(args.damping_factor)  )
        average_hessian = average_hessian +args. damping_factor * torch.eye(average_hessian.size(0), device=average_hessian.device)
    average_hessian = average_hessian + args. damping_factor * torch.eye(average_hessian.size(0), device=average_hessian.device)

    return average_hessian

def compute_gradient_unlearn(args, model, forget_dataset):
    model.train()
    dataloader = DataLoader(DatasetSplit(forget_dataset), batch_size=len(forget_dataset), shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    gradient_unlearn = 0

    for input, label, _ in dataloader:
        input, label = input.to(args.device), label.to(args.device)
        outputs = model(input)
        loss = loss_func(outputs, label)
        loss.backward()
        gradient_unlearn += torch.cat([param.grad.view(-1) for param in model.parameters()])
        model.zero_grad()

    # compute average gradient
    gradient_list = gradient_unlearn 

    return gradient_list


def calc_hessian(args, loss, network_param):
    """
    Compute the complete Hessian matrix of a neural network.

    Args:
        loss: The computed loss value, for example, loss = loss_fn(my_net(x), y)
        network_param: Parameters of the neural network, for example, network_param = my_net.parameters()
        device: Device string, for example, 'cuda' or 'cpu'

    Returns:
        The complete Hessian matrix

    The parameter order of the Hessian matrix is obtained by creating a list [param.flatten() for param in my_net.parameters()].
    """
    torch.cuda.empty_cache()

    param_list = [param.to(args.device) for param in network_param]
    first_derivative = torch.autograd.grad(loss, param_list, create_graph=True)
    derivative_tensor = torch.cat([tensor.flatten().to(args.device) for tensor in first_derivative])
    num_parameters = derivative_tensor.shape[0]
    hessian = torch.zeros(num_parameters, num_parameters).to(args.device)

    for col_ind in range(num_parameters):
        jacobian_vec = torch.zeros(num_parameters, device=args.device)
        jacobian_vec.to(args.device)
        jacobian_vec[col_ind] = 1.
        derivative_tensor.backward(jacobian_vec, retain_graph=True)
        hessian_col = torch.cat([param.grad.flatten().to(args.device) for param in param_list])
        hessian[:, col_ind] = hessian_col
        for param in param_list:
            param.grad.zero_()    


    return hessian






