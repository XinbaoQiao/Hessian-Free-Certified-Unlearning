import os

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import shutil
import numpy as np
import torch as t
from torchvision import transforms



def data_processing_celeba():
    df1 = pd.read_csv('./data/celeba/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male'])
    df1.loc[df1['Male'] == -1, 'Male'] = 0

    df2 = pd.read_csv('./data/celeba/list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
    df2.columns = ['Filename', 'Partition']
    df2 = df2.set_index('Filename')

    df3 = df1.merge(df2, left_index=True, right_index=True)
    # df3.head()


    # df3.to_csv('./data/celeba/celeba-gender-partitions.csv')
    # df4 = pd.read_csv('./data/celeba/celeba-gender-partitions.csv', index_col=0)
    # df4.head()


    # df4.loc[df4['Partition'] == 0].to_csv('./data/celeba/celeba-gender-train.csv')
    # df4.loc[df4['Partition'] == 1].to_csv('./data/celeba/celeba-gender-valid.csv')
    # df4.loc[df4['Partition'] == 2].to_csv('./data/celeba/celeba-gender-test.csv')

    train_samples = 50000  
    test_samples = int(train_samples*0.4)   


    train_df = df3[df3['Partition'] == 0].head(train_samples)
    test_df = df3[df3['Partition'] == 2].head(test_samples)

    valid_df = df3[~df3.index.isin(train_df.index) & ~df3.index.isin(test_df.index)]

    train_df.to_csv('./data/celeba/celeba-gender-train.csv')
    valid_df.to_csv('./data/celeba/celeba-gender-valid.csv')
    test_df.to_csv('./data/celeba/celeba-gender-test.csv')



class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    

class LFWDataSet(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(LFWDataSet, self).__init__()
        self.data = DataArray
        self.label = LabelArray

    def __getitem__(self, index):
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation((0, 30)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return im_trans(self.data[index]), t.tensor(self.label[index], dtype=t.long)

    def __len__(self):
        return self.label.shape[0]



def data_processing_lfw():
    # tar zxvf data/lfw.tgz -C./data
    # src_dir = './data/lfw'
    # dst_dir = './data/lfw_filtered'

    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)

    # for subfolder in os.listdir(src_dir):
    #     subfolder_path = os.path.join(src_dir, subfolder)
    #     if os.path.isdir(subfolder_path):
    #         img_count = len(os.listdir(subfolder_path))
    #         if 30 <= img_count <= 100:
    #             dst_subfolder_path = os.path.join(dst_dir, subfolder)
    #             shutil.copytree(subfolder_path, dst_subfolder_path)
    #         else:
    #             shutil.rmtree(subfolder_path)

    src_dir = './data/lfw'
    dst_dir = './data/lfw_filtered'

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for subfolder in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, subfolder)
        if os.path.isdir(subfolder_path):
            img_count = len(os.listdir(subfolder_path))
            if 30 <= img_count <= 100:
                for img_file in os.listdir(subfolder_path):
                    src_img_path = os.path.join(subfolder_path, img_file)
                    dst_img_path = os.path.join(dst_dir, img_file)
                    shutil.copy(src_img_path, dst_img_path)
            else:
                shutil.rmtree(subfolder_path)

def data_preprocessing_adult(in_path, out_path):
    df = pd.read_csv(in_path)

    df.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True) 
    # print(df.columns)
    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True) 
    # print(df.shape[0])

    new_columns = ['workclass', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']
    for col in new_columns:
        df = df[~df[col].str.contains(r'\?', regex=True)]
    print(df.shape[0])
    print(df.head)

    continuous_column = ['age', 'capitalGain',  'capitalLoss', 'hoursPerWeek']
    age_bins = [0, 23, 28, 33, 38, 43, 53, 63, 100]
    df['age'] = pd.cut(df['age'], age_bins, labels=False)
    capGain_bins = [-1, 100, 1000, 5000, 10000, 30000, 50000, 100000, 150000]
    df['capitalGain'] = pd.cut(df['capitalGain'], capGain_bins, labels=False)
    capLossbins = [-1, 1, 1500, 2000, 4000, 6000, 10000]
    df['capitalLoss'] = pd.cut(df['capitalLoss'], capLossbins, labels=False)
    hourbins = [0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 100]
    df['hoursPerWeek'] = pd.cut(df['hoursPerWeek'], hourbins, labels=False)

    # categorical_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry']
    categorical_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'nativeCountry']
    df = pd.get_dummies(df, columns=categorical_columns)

    income_mapping = {' <=50K': 0, ' <=50K.': 0,' >50K': 1,' >50K.': 1}
    df['income'] = df['income'].map(income_mapping)

    columns_to_drop = ['relationship', 'race', 'sex']  
    df.drop(columns=columns_to_drop, inplace=True)

    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    # data_processing_celeba()
    # data_processing_lfw()

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
        'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
    df_train_set = pd.read_csv('./data/adult/adult_train.txt', names=columns)
    df_test_set = pd.read_csv('./data/adult/adult_test.txt', names=columns, skiprows=1)

    df_train_set.to_csv('./data/adult/train_adult.csv', index=False)
    df_test_set.to_csv('./data/adult/test_adult.csv', index=False)
    data_preprocessing_adult('./data/adult/train_adult.csv','./data/adult/train_adult_processed.csv')
    data_preprocessing_adult('./data/adult/test_adult.csv','./data/adult/test_adult_processed.csv')
