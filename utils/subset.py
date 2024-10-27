import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def reduce_dataset_size(dataset, max_samples,random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    subset_indices = torch.randperm(len(dataset))[:max_samples]
    reduced_dataset = Subset(dataset, subset_indices)
    return reduced_dataset

def sample_dataset_size(dataset, random_seed,indices_to_unlearn):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    reduced_dataset = Subset(dataset, indices_to_unlearn)
    return reduced_dataset


def get_non_overlapping_subsets(dataset, subdataset_train, num_new_data, random_seed):
    subdataset_train_indices = set(subdataset_train.indices)  
    all_indices = set(range(len(dataset)))
    remaining_indices = list(all_indices - subdataset_train_indices)
    random.seed(random_seed)  
    new_data_indices = random.sample(remaining_indices, num_new_data)
    new_data = Subset(dataset, new_data_indices)
    return new_data