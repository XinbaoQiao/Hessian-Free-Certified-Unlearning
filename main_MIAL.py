# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)
import random
import math
import os
import argparse
import numpy as np
import multiprocessing as mp
from torch.nn import functional as F
import torch
from functools import partial
import wandb
import cv2
import copy
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from utils.options import args_parser
from utils.subset import reduce_dataset_size,get_non_overlapping_subsets
from utils.perturbation import NoisedNetReturn
import utils.loading_data as dataset
from models.Nets import MLP, CNNMnist, CNNCifar, Logistic, LeNet, FashionCNN4
from tqdm import tqdm
import functools
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
import scipy.stats
import time
from sklearn.metrics import auc, roc_curve



###############################################################################
#                               SHADOW                                        #
###############################################################################

def run(dataset_train,dataset_test,net_shadow,shadow_id):
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    args.shadow_id = shadow_id
    args.debug = True
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)


    size = len(dataset_train)
    # np.random.seed(seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True
    dataset_train = torch.utils.data.Subset(dataset_train, keep)
    train_dl = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(dataset_test, batch_size=2048, shuffle=False, num_workers=4)

    # Model
    m = net_shadow.to(args.device)

    optim = torch.optim.SGD(m.parameters(), lr=args.MIAlr,  weight_decay=args.regularization)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.lr_decay)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.MIAepochs)

    # Train
    for i in range(args.MIAepochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(args.device), y.to(args.device)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        wandb.log({"loss": loss_total / len(train_dl)})
    # print('Seed',seed)
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    wandb.log({"acc_test": get_acc(m, test_dl)})

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(args.device), y.to(args.device)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


###############################################################################
#                               QUERY                                        #
###############################################################################

def query(args, net, logitspath,train_shadow, train_dl):
    net=net.to(args.device)
    for path in os.listdir(args.savedir):
        current_path = os.path.join(args.savedir, path)
        current_logitspath = os.path.join(current_path, logitspath)  
        net_shadow = copy.deepcopy(net)
        net_shadow.load_state_dict(torch.load(os.path.join(current_path, "model.pt")))    
        logits_n_shadow = []
        for i in range(args.n_queries):
            logits_shadow = []
            for x, _ in tqdm(train_shadow, desc=f"Processing query {i+1} for net_shadow"):
                x = x.to(args.device)
                outputs_shadow = net_shadow(x)
                logits_shadow.append(outputs_shadow.detach().cpu().numpy())
            logits_n_shadow.append(np.concatenate(logits_shadow, axis=0))
        logits_n_shadow = np.stack(logits_n_shadow, axis=1)  # Shape: (n_samples, n_queries, n_classes)
        # print(logits_n_shadow.shape)
        logits_n_net = []
        for x, _ in tqdm(train_dl, desc="Processing for net"):
            x = x.to(args.device)
            outputs_net = net(x)
            logits_n_net.append(outputs_net.detach().cpu().numpy())
        logits_n_net = np.concatenate(logits_n_net, axis=0)  # Shape: (n_samples, n_classes)
        logits_n_net = np.expand_dims(logits_n_net, axis=1)  # Shape: (n_samples, 1, n_classes)
        # logits_n = np.concatenate([logits_n_shadow, logits_n_net], axis=1)  # Shape: (n_samples, n_queries + 1, n_classes)
        # print(logits_n.shape)
        np.save(os.path.join(current_path, "logits.npy") , logits_n_shadow)
        np.save(current_logitspath , logits_n_net )

###############################################################################
#                               ATTACK                                        #
###############################################################################

def load_one(path,dataset_train,logitspath,scorespath,label=None):
    """
    This loads a logits and converts it to a scored prediction.
    """
    opredictions = np.load(os.path.join(path, logitspath))  # [n_examples, n_augs, n_classes]

    # Be exceptionally careful.
    # Numerically stable everything, as described in the paper.
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    # print(predictions)
    if label == None:
        labels = np.array(dataset_train.targets)
    else:
        labels0 = np.array(dataset_train.datasets[0].dataset.targets)
        labels1 = np.array(dataset_train.datasets[1].dataset.targets)
        labels = np.concatenate([labels0, labels1])

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    np.save(os.path.join(path,scorespath), logit)

def load_stats(dataset_train,logitspath,scorespath,label=None):
    load_one_with_dataset = partial(load_one, dataset_train=dataset_train,logitspath=logitspath,scorespath=scorespath,label=label)
    
    with mp.Pool(8) as p:
        paths = [os.path.join(args.savedir, x) for x in os.listdir(args.savedir)]
        p.map(load_one_with_dataset, paths)




###############################################################################
#                               PLOT                                          #
###############################################################################

def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data(scorespath,keeppath,label=None):
    """
    Load our saved scores and then put them into a big matrix.
    """
    # global scores, keep
    scores = []
    keep = []
    if label=='shadow':
        for path in os.listdir(args.savedir):
            scores.append(np.load(os.path.join(args.savedir, path, scorespath)))
            keep.append(np.load(os.path.join(args.savedir, path, keeppath)))
        scores = np.array(scores)
        keep = np.array(keep)
    else:
        scores.append(np.load(os.path.join(args.savedir, '0', scorespath)))
        keep.append(np.load(os.path.join(args.savedir, '0', keeppath)))
        scores = np.array(scores)
        keep = np.array(keep)
    print(scores.shape)
    print(keep.shape)
    return scores, keep


def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []
    print(scores.shape)
    print(keep.shape)
    print(check_keep.shape)
    print(check_scores.shape)

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []
    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(fn, keep, scores, ntest, keep_target, scores_target, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    # prediction, answers = fn(keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:])
    prediction, answers = fn(keep, scores,  keep_target , scores_target)

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    # low = tpr[np.where(fpr < 0.01)[0][-1]]   ## 1% FPR

    # print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.4f" % auc
    elif metric == "acc":
        metric_text = "acc=%.4f" % acc  

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr(scores, keep, scores_list,keep_list,labels):
    plt.figure(figsize=(4, 3))

    for scores_target, keep_target, label in zip(scores_list, keep_list, labels):
        do_plot(functools.partial(generate_ours), keep, scores, 1,keep_target, scores_target,label + "\n", metric="auc")

    # do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")
    # do_plot(functools.partial(generate_ours, fix_variance=True), keep, scores, 1, "Ours (online, fixed variance)\n", metric="auc")
    # do_plot(functools.partial(generate_ours_offline), keep, scores, 1, "HF\n", metric="AUC")
    # do_plot(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, 1, "Ours (offline, fixed variance)\n", metric="auc")

    # do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

    # plt.semilogx()
    # plt.semilogy()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    rootpath = './results/MIAL/'
    filename = 'MIAL_model_{}_data_{}_remove_{}_epoch_{}_seed{}_validation_{}_std_{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs,args.seed,args.application, args.std)
    output_file_path = os.path.expanduser(os.path.join(rootpath, filename))
    os.makedirs(rootpath, exist_ok=True)
    plt.savefig(output_file_path)


if __name__ == '__main__':
    dir_path = './MIAL'

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Successfully deleted the directory: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")

    ###################### SETUP ######################
    args = args_parser()
    args.n_queries = 2
    args.MIAepochs = 30
    args.n_shadows = 16
    args.MIAlr = 0.05
    args.pkeep = 0.5
    args.savedir = "MIAL/Shadow"
    args.debug = False 

    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    path2="./data"
    if not os.path.exists(path2):
        os.makedirs(path2)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    def filter_indices(dataset, target_classes):
        return [i for i, target in enumerate(dataset.targets) if target.item() in target_classes]
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
    
    elif args.dataset == 'fer':
        data_dir = './data/fer2013/'
        normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        # define transforms
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.2),
            normalize,
        ])
        dataset_train  = datasets.ImageFolder(root=data_dir+'train',transform=transform)
        dataset_test = datasets.ImageFolder(root=data_dir+'test',transform=transform)
    elif args.dataset == 'cifar':

        transform = transforms.Compose([transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])     
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    elif args.dataset == 'celeba':
        args.num_classe = 2
        args.bs = 64
        custom_transform =transforms.Compose([
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])

        transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
                               )

        dataset_train = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-train.csv',
                                      img_dir='./data/celeba/img_align_celeba/',
                                      transform=custom_transform)
        # valid_celeba= dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-valid.csv',
        #                               img_dir='./data/celeba/img_align_celeba/',
        #                               transform=custom_transform)
        dataset_test = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-test.csv',
                                     img_dir='./data/celeba/img_align_celeba/',
                                     transform=custom_transform)
    elif args.dataset == 'svhn':
        num_classes = 10
        train_transform = transforms.Compose([])
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        dataset_train = datasets.SVHN(root='./data/svhn',
                                    split='train',
                                    transform=train_transform,
                                    download=True)

        dataset_test = datasets.SVHN(root='./data/svhn',
                                    split='test',
                                    transform=train_transform,
                                    download=True)
    elif args.dataset == 'lfw':
        num_classes = 29
        path = './data/lfw'
        pathlist = map(lambda x: '/'.join([path, x]), os.listdir(path))
        namedict = {}
        data, label = [], []
        idx = 0
        for item in pathlist:
            dirlist = os.listdir(item)
            if not (30<= len(dirlist) <= 100):
                continue
            for picpath in dirlist:
                data.append(cv2.imread(item + '/' + picpath))
                label.append(idx)
            namedict[str(idx)] = item.split('/')[-1]
            idx += 1
        data, label = np.stack(data), np.array(label)
        idx = np.random.permutation(data.shape[0])
        data, label = data[idx], label[idx]
        train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)
        args.test_train_rate = 1; args.epochs-=1; args.batch_size+=1
        dataset_train = dataset.LFWDataSet(train_X, train_Y)
        dataset_test= dataset.LFWDataSet(test_X, test_Y)
        args.num_dataset = len(dataset_train)
    else:
        exit('Error: unrecognized dataset')

    
    subdataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    subdataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    img_size = dataset_train[0][0].shape
    print('Train dataset:   ',len(dataset_train))
    print('Test dataset:   ',len(dataset_test))

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
        for param in net.parameters():
            param.requires_grad = False
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
    print(net)
    net.eval()
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total)," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))

    Original_logitspath = "Original_logits.npy"
    Proposed_logitspath = "Proposed_logits.npy"
    IJ_logitspath =  "IJ_logits.npy"
    NU_logitspath =  "NU_logits.npy"
    Retrain_logitspath = "Retrain_logits.npy"

    Original_scorespath = "Original_scores.npy"
    Proposed_scorespath = "Proposed_scores.npy"
    IJ_scorespath = "IJ_scores.npy"
    NU_scorespath =  "NU_scores.npy"
    Retrain_scorespath = "Retrain_scores.npy"

    net_target_Proposed = copy.deepcopy(net)
    net_target_Retrain = copy.deepcopy(net)
    net_target_Original = copy.deepcopy(net)
    net_target_IJ = copy.deepcopy(net)
    net_target_NU = copy.deepcopy(net)

    net.train()
    net_shadow = copy.deepcopy(net)

###############################################################################
#                               Shadow                                        #
###############################################################################  
    all_indices = list(range(len(subdataset_train)))
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    remaining_indices = list(set(all_indices) - set(indices_to_unlearn))
    forget_dataset = Subset(subdataset_train, indices_to_unlearn)
    remain_dataset = Subset(subdataset_train, remaining_indices)

    new_data = get_non_overlapping_subsets(dataset_train, subdataset_train,len(dataset_train) - len(subdataset_train) , args.seed)
    combined_data = ConcatDataset([subdataset_train, new_data])
    keep_bool = np.full(len(combined_data), False)
    keep_bool[:len(subdataset_train)] = True
    train_keep_path ="train_keep.npy"
    train_dl = DataLoader(combined_data, batch_size=args.batch_size, shuffle=False)

    # new_data = get_non_overlapping_subsets(dataset_train, subdataset_train, args.num_dataset *3,  np.random.randint(0, 1000000000))
    shadowdataset_train = ConcatDataset([subdataset_train, new_data])
    for shadow_id in range(args.n_shadows):
        run(shadowdataset_train,dataset_test,net_shadow=copy.deepcopy(net_shadow), shadow_id=shadow_id)
        savedir = os.path.join(args.savedir, str(args.shadow_id))
        np.save(os.path.join(savedir,train_keep_path) , keep_bool)
    train_shadow = DataLoader(shadowdataset_train, batch_size=args.batch_size, shuffle=False)

###############################################################################
#                               QUERY                                         #
###############################################################################
    rootpath = './log'
    Original_model_path = rootpath + '/Original/Model/Original_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model, args.dataset, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    Original_model_state_dict = torch.load(Original_model_path)
    net_target_Original.load_state_dict(Original_model_state_dict)
    # if args.application ==True:  
    #     w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=1, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
    #     net_target_Original.load_state_dict(w)
    query(args,net_target_Original,Original_logitspath,train_shadow,train_dl)

    Proposed_model_path = rootpath + '/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    Proposed_model_state_dict = torch.load(Proposed_model_path)
    net_target_Proposed.load_state_dict(Proposed_model_state_dict)
    if args.application ==True:  
        args.std = 0.0025  ## ε_LR ≈ 2
        # args.std = 0.0265   ## ε_CNN ≈ 120
        w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=1, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
        net_target_Proposed.load_state_dict(w)
    query(args,net_target_Proposed,Proposed_logitspath,train_shadow,train_dl)


    IJ_model_path = rootpath + '/IJ/Model/IJ_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    IJ_model_state_dict = torch.load(IJ_model_path)
    net_target_IJ.load_state_dict(IJ_model_state_dict)
    query(args,net_target_IJ,IJ_logitspath,train_shadow,train_dl)
    # if args.application ==True:  
    #     w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=1, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
    #     net_target_IJ.load_state_dict(w)

    NU_model_path = rootpath + '/NU/Model/NU_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    NU_model_state_dict = torch.load(NU_model_path)
    net_target_NU.load_state_dict(NU_model_state_dict)
    # if args.application ==True:  
    #     w = NoisedNetReturn(args, net=copy.deepcopy(net).to(args.device), rho=1, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
    #     net_target_NU.load_state_dict(w)
    query(args,net_target_NU,NU_logitspath,train_shadow,train_dl)

    Retrain_model_path = rootpath + '/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    Retrain_model_state_dict = torch.load(Retrain_model_path)
    net_target_Retrain.load_state_dict(Retrain_model_state_dict)
    query(args,net_target_Retrain,Retrain_logitspath,train_shadow,train_dl)


###############################################################################
#                               ATTACK                                        #
###############################################################################
    load_stats(shadowdataset_train,"logits.npy","scores.npy",label='flag')
    load_stats(combined_data,Original_logitspath,Original_scorespath,label='flag')
    load_stats(combined_data,Proposed_logitspath,Proposed_scorespath,label='flag')
    load_stats(combined_data,IJ_logitspath,IJ_scorespath,label='flag')
    load_stats(combined_data,NU_logitspath,NU_scorespath,label='flag')
    load_stats(combined_data,Retrain_logitspath,Retrain_scorespath,label='flag')

###############################################################################
#                               PLOT                                          #
###############################################################################

    scores, keep = load_data("scores.npy","keep.npy",label='shadow')
    Original_scores, Original_keep = load_data(Original_scorespath,train_keep_path)
    Proposed_scores, Proposed_keep = load_data(Proposed_scorespath,train_keep_path)
    IJ_scores, IJ_keep = load_data(IJ_scorespath,train_keep_path)
    NU_scores, NU_keep = load_data(NU_scorespath,train_keep_path)
    Retrain_scores, Retrain_keep = load_data(Retrain_scorespath,train_keep_path)

    scores_list = [Original_scores,Proposed_scores, IJ_scores, NU_scores, Retrain_scores]
    keep_list = [Original_keep,Proposed_keep, IJ_keep, NU_keep, Retrain_keep]
    labels = ['Original', 'Proposed','IJ', 'NU',  'Retrain']
    fig_fpr_tpr( scores, keep, scores_list, keep_list,labels)
