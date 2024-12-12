import random
import time
import math
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
from utils.options import args_parser
from utils.subset import reduce_dataset_size   

def load_dataset():
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

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
    elif args.dataset == 'svhn':
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

    elif args.dataset == 'adult':  
        args.num_classes = 2
        class Adult(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)  
                        
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                if self.transform:
                    sample = self.transform(sample)
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long)  # Convert labels to long

        train_data = pd.read_csv("./data/adult/train_adult_processed.csv")
        test_data = pd.read_csv("./data/adult/test_adult_processed.csv")
        pd.set_option("display.max_columns", 102)
        common_cols = set(train_data.columns).intersection(test_data.columns)
        train_feature = train_data[list(common_cols - {'income'})]
        train_target = train_data['income']
        test_feature = test_data[list(common_cols - {'income'})]
        test_target = test_data['income']

        x_train, x_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=args.test_train_rate, random_state=args.seed)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train) 
        x_test = scaler.transform(x_test)        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) 
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)   

        dataset_train = Adult(x_train_tensor, y_train_tensor)
        dataset_test = Adult(x_test_tensor, y_test_tensor)

    elif args.dataset == 'creditcard':
        args.num_classes = 2
        class CreditCard(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                
                if self.transform:
                    sample = self.transform(sample)
                
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long)  # Convert labels to long

        data = pd.read_csv("./data/creditcard/creditcard.csv")
        data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
        data = data.drop(['Time', 'Amount'], axis=1)

        X = data.loc[:, data.columns != 'Class']
        Y = data.loc[:, data.columns == 'Class'].squeeze()  # Convert to 1D array

        # Apply under-sampling as you did
        number_records_fraud = len(data[data.Class == 1])
        fraud_indices = np.array(data[data.Class == 1].index)
        normal_indices = data[data.Class == 0].index
        random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
        random_normal_indices = np.array(random_normal_indices)
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

        under_sample_data = data.iloc[under_sample_indices, :]
        X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
        Y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class'].squeeze()

        # Showing ratio
        print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
        print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
        print("Total number of transactions in resampled data: ", len(under_sample_data))

        # Whole dataset split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.test_train_rate, random_state=args.seed)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        dataset_train = CreditCard(x_train_tensor, y_train_tensor)
        dataset_test = CreditCard(x_test_tensor, y_test_tensor)

    elif args.dataset == 'cancer':
        args.num_classes = 2
        class Cancer(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                
                if self.transform:
                    sample = self.transform(sample)
                
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long)  
            
        breast = pd.read_csv("./data/cancer/breast-cancer.csv")
        breast["diagnosis"] = breast["diagnosis"].map({"M": 1, "B": 0})
        train_data, test_data  = train_test_split(breast, test_size=args.test_train_rate, random_state=args.seed)
        x_train = train_data.drop(columns=["diagnosis", "id"])
        x_test = test_data.drop(columns=["diagnosis", "id"])
        y_train = train_data["diagnosis"]
        y_test = test_data["diagnosis"]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        dataset_train = Cancer(x_train_tensor, y_train_tensor)
        dataset_test = Cancer(x_test_tensor, y_test_tensor)

    elif args.dataset == 'wine':
        args.num_classes = 3
        class Wine(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                
                if self.transform:
                    sample = self.transform(sample)
                
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long) 
        data = load_wine()
        X = data.data # Sample 178
        y = data.target # Class 0,1,2

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed) # 142 Training sample，36 Testing sample

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        dataset_train = Wine(x_train_tensor, y_train_tensor)
        dataset_test = Wine(x_test_tensor, y_test_tensor)

    elif args.dataset == 'hapt':
        args.num_classes = 6
        class HAPT(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                
                if self.transform:
                    sample = self.transform(sample)
                
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long) 
        def datafile_load(file_path):
            data = pd.read_table(file_path, delim_whitespace=True,header=None)
            return data
        x_train = datafile_load("./data/hapt/train/X_train.txt")
        y_train = datafile_load("./data/hapt/train/y_train.txt")
        x_test = datafile_load("./data/hapt/test/X_test.txt")
        y_test = datafile_load("./data/hapt/test/y_test.txt")
        print(y_train.shape)  # 检查 y_train 的形状

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.long)
        dataset_train = HAPT(x_train_tensor, y_train_tensor)
        dataset_test = HAPT(x_test_tensor, y_test_tensor)

    elif args.dataset == 'obesity':
        args.num_classes = 7
        class Obesity(Dataset):
            def __init__(self, data, targets, transform=None):
                if data is None or targets is None:
                    raise ValueError("Data or targets cannot be None.")
                self.data = data
                self.targets = targets
                self.transform = transform
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                target = self.targets[idx]
                
                if self.transform:
                    sample = self.transform(sample)
                
                return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.long) 
        df=pd.read_csv("./data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv")
        # Data Preprocessing
        clean_df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
        clean_df = pd.get_dummies(clean_df, columns=['family_history_with_overweight'], drop_first=True)

        clean_df = pd.get_dummies(clean_df, columns=['FAVC'], drop_first=True)

        clean_df['CAEC'].replace(['Sometimes', 'Frequently', 'Always', 'no'], [1, 2, 3, 0], inplace=True)
        clean_df['CALC'].replace(['no', 'Sometimes', 'Frequently', 'Always'], [0, 1, 2, 3], inplace=True)

        clean_df = pd.get_dummies(clean_df, columns=['SMOKE'], drop_first=True)
        clean_df = pd.get_dummies(clean_df, columns=['SCC'], drop_first=True)
        clean_df = pd.get_dummies(clean_df, columns=['MTRANS'], drop_first=True)

        X = clean_df.drop('NObeyesdad', axis=1)
        y = clean_df['NObeyesdad']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        y_train = label_encoder.fit_transform(y_train)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        dataset_train = Obesity(x_train_tensor, y_train_tensor)
        dataset_test = Obesity(x_test_tensor, y_test_tensor)


    # elif args.dataset == 'cifar':

    #     transform = transforms.Compose([transforms.ToTensor(),
    #        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])     
    #     dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
    #     dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    # elif args.dataset == 'celeba':
    #     args.num_classe = 2
    #     args.bs = 64
    #     custom_transform =transforms.Compose([
    #                                             transforms.Resize((128, 128)),
    #                                             transforms.ToTensor(),
    #                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                                         ])

        # transform = transforms.Compose([transforms.Resize((224, 224)),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
        #                        )

    #     dataset_train = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-train.csv',
    #                                   img_dir='./data/celeba/img_align_celeba/',
    #                                   transform=custom_transform)
    #     # valid_celeba= dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-valid.csv',
    #     #                               img_dir='./data/celeba/img_align_celeba/',
    #     #                               transform=custom_transform)
    #     dataset_test = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-test.csv',
    #                                  img_dir='./data/celeba/img_align_celeba/',
    #                                  transform=custom_transform)
    # elif args.dataset == 'lfw':
    #     num_classes = 29
    #     path = './data/lfw'
    #     pathlist = map(lambda x: '/'.join([path, x]), os.listdir(path))
    #     namedict = {}
    #     data, label = [], []
    #     idx = 0
    #     for item in pathlist:
    #         dirlist = os.listdir(item)
    #         if not (30<= len(dirlist) <= 100):
    #             continue
    #         for picpath in dirlist:
    #             data.append(cv2.imread(item + '/' + picpath))
    #             label.append(idx)
    #         namedict[str(idx)] = item.split('/')[-1]
    #         idx += 1
    #     data, label = np.stack(data), np.array(label)
    #     idx = np.random.permutation(data.shape[0])
    #     data, label = data[idx], label[idx]
    #     train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)
    #     args.test_train_rate = 1; args.epoch-=1; args.batch_size+=1
    #     dataset_train = dataset.LFWDataSet(train_X, train_Y)
    #     dataset_test= dataset.LFWDataSet(test_X, test_Y)
        # args.num_dataset = len(dataset_train)
    else:
        exit('Error: unrecognized dataset')

    
    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    print('Train dataset:   ',len(dataset_train))
    print('Test dataset:   ',len(dataset_test))

    return dataset_train, dataset_test, args.num_classes