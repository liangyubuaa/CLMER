import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .augmentations import DataTransform
import pickle

class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

def data_generator(data_path, configs, training_mode, dataset, modality):
    if "self_supervised" in training_mode:
        preprocess(data_path, dataset, modality)
    batch_size = configs.batch_size
    print(os.path.join(data_path, f"{modality}train.pt"))
    train_dataset = torch.load(os.path.join(data_path, f"{modality}train.pt"))
        
    
    valid_dataset = torch.load(os.path.join(data_path, f"{modality}val.pt"))
    test_dataset = torch.load(os.path.join(data_path, f"{modality}test.pt"))
    
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    
    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader
def preprocess(data_path, dataset, modality):
    origin_path = f'{data_path}/{dataset}.pkl'
    origin = pickle.load(open(origin_path, 'rb'))
    print(origin['train'][modality].shape)
    
    train_pt = {
        "samples":origin['train'][modality].transpose(0,2,1),
        "labels":origin['train']['labels'].squeeze().numpy()
        }
    test_pt = {
        "samples":origin['test'][modality].transpose(0,2,1),
        "labels":origin['test']['labels'].squeeze().numpy()
        }
    valid_pt = {
        "samples":origin['valid'][modality].transpose(0,2,1),
        "labels":origin['valid']['labels'].squeeze().numpy()
        }

    torch.save(train_pt, os.path.join(data_path, f"{modality}train.pt"))
    torch.save(test_pt, os.path.join(data_path, f"{modality}test.pt"))
    torch.save(valid_pt, os.path.join(data_path, f"{modality}val.pt"))
    print(train_pt['samples'].shape)
    print(train_pt['labels'].shape)
    print(valid_pt['samples'].shape)
    print(valid_pt['labels'].shape)
    print(test_pt['samples'].shape)
    print(test_pt['labels'].shape)
    print("End")