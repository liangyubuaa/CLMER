import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='deap', split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))
        print(dataset.keys())
        # print(dataset['train'])
        self.vision1 = torch.tensor(dataset[split_type]['vision1'].astype(np.float32)).cpu().detach()
        self.physio = torch.tensor(dataset[split_type]['physio'].astype(np.float32)).cpu().detach()
        self.vision2 = dataset[split_type]['vision2'].astype(np.float32)
        self.vision2[self.vision2 == -np.inf] = 0
        self.vision2 = torch.tensor(self.vision2).cpu().detach()
        data = data[:-1]
        self.labels = torch.tensor(dataset[split_type]['labels']).cpu().detach()
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        
        self.data = data
        
        self.n_modalities = 3 
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.physio.shape[1], self.vision2.shape[1], self.vision1.shape[1]
    def get_dim(self):
        return self.physio.shape[2], self.vision2.shape[2], self.vision1.shape[2]
    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.physio[index], self.vision2[index], self.vision1[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        return X, Y, META