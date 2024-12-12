# Python version: 3.6
import random
import math
import os
import cv2
import copy
import numpy as np
import torch
import shutil
import matplotlib
import torch.nn as nn
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score ,roc_curve,auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet18
from utils.Approximator import getapproximator
from utils.Approximator_resnet import getapproximator_resnet
from utils.options import args_parser
from utils.subset import reduce_dataset_size
from utils.perturbation import NoisedNetReturn
import utils.loading_data as dataset
from models.Update import train
from models.Nets import MLP, CNNMnist, CNNCifar, Logistic, LeNet, FashionCNN4
from models.test import test_img, test_per_img



class MLP_INF:
    def __init__(self):
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])

    def test_model_roc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        idx = np.where(fpr <= 0.01)[0][-1]
        tpr_at_1_percent_fpr = tpr[idx]
        low  = tpr_at_1_percent_fpr * 100
        return fpr, tpr, low

class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400, multi_class='ovr', n_jobs=1)

    def train_model(self, train_x, train_y):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        # temperature = 1
        # train_x /= temperature
        self.model.fit(self.scaler.transform(train_x), train_y)

    def predict_proba(self, test_x):
        self.scaler = preprocessing.StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        # return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC

    def test_model_roc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        idx = np.where(fpr <= 0.01)[0][-1]
        tpr_at_1_percent_fpr = tpr[idx]
        low  = tpr_at_1_percent_fpr * 100
        return fpr, tpr, low

    


def posterior(dataloader, model, args):
    posterior_list = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(args.device), labels.to(args.device)
            outputs = model(data)
            post = torch.softmax(outputs, 1)
            posterior_list.append(post)
    return torch.cat(posterior_list, 0)


def construct_feature(post_ori, post_unl, method):
    if method == "direct_diff":
        return post_ori - post_unl

    elif method == "sorted_diff":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return post_ori - post_unl

    elif method == "l2_distance":
        feat = torch.ones(post_ori.shape[0])
        for index in range(post_ori.shape[0]):
            euclidean = distance.euclidean(post_ori[index], post_unl[index])
            feat[index] = euclidean
        return feat.unsqueeze(1)

    elif method == "direct_concat":
        return torch.cat([post_ori, post_unl], 1)

    elif method == "sorted_concat":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return torch.cat([post_ori, post_unl], 1)

def attack(args,net_target,net_learned,unl_loader,res_loader, dataset_test, unlearn_method):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    net_learned=net_learned.to(args.device); net_target=net_target.to(args.device)
    ori_pos_post = posterior(unl_loader, net_learned, args).detach().cpu()
    ori_neg_post = posterior(res_loader, net_learned, args).detach().cpu()
    unl_pos_post = posterior(unl_loader , net_target, args).detach().cpu()
    unl_neg_post = posterior(res_loader, net_target, args).detach().cpu()
    feat_pos = construct_feature(ori_pos_post, unl_pos_post, args.method)
    feat_neg = construct_feature(ori_neg_post, unl_neg_post, args.method)
    print('Dif:' , ori_pos_post-unl_pos_post)
    feat = torch.cat([feat_pos, feat_neg], 0).numpy()
    label = torch.cat([torch.ones(feat_pos.shape[0]), torch.zeros(feat_neg.shape[0])], 0).numpy().astype('int')
    if args.attack_model == 'LR':
        attack_model = LR()
    elif args.attack_model == 'MLP':
        attack_model = MLP_INF()
    else:
        raise Exception("invalid attack name")
    attack_model.train_model(feat, label)
    train_auc = attack_model.test_model_auc(feat, label)
    fpr, tpr, low =  attack_model.test_model_roc(feat, label)
    acc, _  = test_img(net_target, dataset_test, args)
    print(f"Target Model: {unlearn_method}, Attack AUC: {100 * train_auc:.2f}%, TPR@1%FPR : {low:.5f}%")   
    return fpr, tpr, train_auc,acc, low


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
    else:
        exit('Error: unrecognized dataset')

    
    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    img_size = dataset_train[0][0].shape
    print('Train dataset:   ',len(dataset_train))
    print('Test dataset:   ',len(dataset_test))

    net = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset != 'cifar':
        net = CNNMnist(args=args).to(args.device)
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
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total)," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))

    net_learned = copy.deepcopy(net)
    net_target_Proposed = copy.deepcopy(net)
    net_target_Proposed2 = copy.deepcopy(net)
    net_target_Proposed3 = copy.deepcopy(net)
    net_target_Retrain = copy.deepcopy(net)
    net_target_Original = copy.deepcopy(net)
    net_target_IJ = copy.deepcopy(net)
    net_target_NU = copy.deepcopy(net) 

    rootpath = './log'
    Original_model_path = rootpath + '/Original/Model/Original_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model,args.dataset,args.epochs,args.lr,args.lr_decay,args.clip,args.seed)
    Proposed_model_path = rootpath + '/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model, args.dataset, args.num_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
    # IJ_model_path = rootpath + '/IJ/Model/IJ_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    # NU_model_path = rootpath + '/NU/Model/NU_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed)
    Retrain_model_path = rootpath + '/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.lr,args.lr_decay,args.clip,args.seed)

    net_learned.load_state_dict(torch.load(Original_model_path))
    net_target_Retrain.load_state_dict(torch.load(Retrain_model_path))
    # net_target_IJ.load_state_dict(torch.load(IJ_model_path))
    # net_target_NU.load_state_dict(torch.load(NU_model_path))
    net_target_Proposed.load_state_dict(torch.load(Proposed_model_path))
    if args.application ==True:  
        w = NoisedNetReturn(args, net=copy.deepcopy(net_target_Proposed).to(args.device), rho=1, epsilon=args.epsilon, delta=args.delta, n=args.num_dataset, m=1)
        net_target_Proposed.load_state_dict(w)

    all_indices = list(range(len(dataset_train)))
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)
    remaining_indices = list(set(all_indices) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)

    unl_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    res_loader = torch.utils.data.DataLoader(remain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    fpr_Retrain, tpr_Retrain, train_auc_Retrain,acc_Retrain, low_Retrain = attack(args,net_target_Retrain,net_learned,unl_loader,res_loader, dataset_test,unlearn_method='Retrain')
    fpr_Proposed, tpr_Proposed, train_auc_Proposed,acc_Proposed,low_Proposed = attack(args,net_target_Proposed,net_learned,unl_loader,res_loader, dataset_test,unlearn_method='Proposed (ε=∞)')
    # fpr_IJ, tpr_IJ, train_auc_IJ,acc_IJ, low_IJ = attack(args,net_target_IJ,net_learned,unl_loader,res_loader, dataset_test,unlearn_method='IJ')
    # fpr_NU, tpr_NU, train_auc_NU,acc_NU, low_NU= attack(args,net_target_NU,net_learned,unl_loader,res_loader, dataset_test,unlearn_method='NS')


    rootpath = './results/MIAU/'
    os.makedirs(rootpath, exist_ok=True)
    filename = 'MIAU_model_{}_data_{}_remove_{}_epoch_{}_seed{}_validation_{}_std_{}.txt'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.seed, args.application, args.std
    )
    file_path = os.path.join(rootpath, filename)
    with open(file_path, 'w') as file:
        # file.write(f"IJ - AUC: {100 * train_auc_IJ:.2f}%, Accuracy: { acc_IJ:.2f}%\n")
        # file.write(f"NS - AUC: {100 * train_auc_NU:.2f}%, Accuracy: {acc_NU:.2f}%\n")
        file.write(f"Retrain - AUC: {100 * train_auc_Retrain:.2f}%, Accuracy: { acc_Retrain:.2f}%\n")
        file.write(f"Proposed (ε=∞) - AUC: {100 * train_auc_Proposed:.2f}%, Accuracy: {acc_Proposed:.2f}%\n")



    color_Retrain = "#F7A1A1" ; color_Proposed = "#D22027" ; color_IJ = "#385989" ;color_NU = "#7FA5B7" ; color_diagonal = "#BEBEBE" 
    plt.figure()
    plt.figure(figsize=(8.5,  7)); plt.subplots_adjust(left=0.175, right=0.925, top=0.925, bottom=0.125)
    # plt.plot(fpr_IJ, tpr_IJ, color=color_IJ, lw=2, label=f'IJ (AUC = {train_auc_IJ:.2f})')
    # plt.plot(fpr_NU, tpr_NU, color=color_NU, lw=2, label=f'NS (AUC = {train_auc_NU:.2f})')
    plt.plot(fpr_Retrain, tpr_Retrain, color=color_Retrain, lw=2, label=f'Retrain (AUC = {train_auc_Retrain:.2f})')
    plt.plot(fpr_Proposed, tpr_Proposed, color=color_Proposed, lw=2, label=f'Proposed  (AUC = {train_auc_Proposed:.2f})')
    plt.plot([0, 1], [0, 1], color=color_diagonal, lw=2, linestyle='--')

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xticks(fontsize=23)  
    plt.yticks(fontsize=23)
    plt.xlabel('False Positive Rate', fontsize=30, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=30, fontweight='bold')
    plt.title('Target Model: {}'.format(args.model), fontsize=30, fontweight='bold')
    # plt.title('Feature Construction: {}'.format(args.method), fontsize=25, fontweight='bold')
    plt.legend(loc="lower right")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15.5, fontweight='bold')

    filename = 'MIAU_model_{}_data_{}_remove_{}_epoch_{}_seed{}_validation_{}_std_{}.png'.format(
        args.model, args.dataset, args.num_forget, args.epochs, args.seed, args.application, args.std
    )
    plt.savefig(os.path.join(rootpath, filename))
    
