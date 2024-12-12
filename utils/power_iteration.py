import numpy as np
import torch
import random
import torch.nn.functional as F


def spectral_radius(args,loss_batch, net,t):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    grad_params = torch.autograd.grad(loss_batch, net.parameters(), create_graph=True, retain_graph=True)
    grad_norm = torch.norm(torch.cat([grad.view(-1) for grad in grad_params]))
    # print('Grad Norm:',grad_norm)
    if grad_norm > args.clip:
        scaling_factor = args.clip / grad_norm
        grad_params = [grad * scaling_factor for grad in grad_params]
    v = [torch.ones_like(param) for param in net.parameters()] 
    v = [F.normalize(tensor, p=2, dim=0) for tensor in v]
    v_old = torch.cat([vec.reshape(-1) for vec in v])

    params = list(net.parameters())
    # adjusted_params = [param - args.lr * grad_param for param, grad_param in zip(params, grad_params)]
    # adjusted_params = [args.lr * (args.lr_decay**t) * grad_param for grad_param in  grad_params]
    adjusted_params = [grad / args.batch_size for grad in grad_params]

    e_value = 0
    for i in range(1000):
        u = torch.autograd.grad(adjusted_params , net.parameters(), grad_outputs=v, retain_graph=True)
        u_flat = torch.cat([grad.reshape(-1) for grad in u])
        grad_norm = torch.norm(torch.cat([grad.reshape(-1) for grad in u]))
        v = [grad / grad_norm for grad in u]  
        v_flat = torch.cat([vec.reshape(-1) for vec in v])
        # Compute eigenvalue approximation
        e_value = torch.dot(v_flat,u_flat)
        tol = torch.norm(v_old  -  v_flat)
        if tol < 1e-7:
            break
        i += 1; v_old =  v_flat
    
    return e_value
