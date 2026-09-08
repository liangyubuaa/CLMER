import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import pickle
import os
from scipy import signal
import torch
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor):
        self.images = image_tensor  # (N, 3, 112, 112)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='deap', split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        self.vision1 = torch.tensor(dataset[split_type]['vision1'].astype(np.float32)).cpu().detach()
        #################################MobileViT###########################################
        # N = self.vision1.shape[0]
        # self.vision1 = self.vision1.view(N, 112, 112, 3)              # [N, 112, 112, 3]
        # self.vision1 = self.vision1.permute(0, 3, 1, 2)               # [N, 3, 112, 112]
        # self.vision1 = F.interpolate(self.vision1, size=(32, 32), mode='bilinear', align_corners=False) # [N, 3, 56, 56]
        # self.vision1 = self.vision1.permute(0, 2, 3, 1)               # [N, 56, 56, 3]
        # self.vision1 = self.vision1.reshape(N, 32*32, 3)              # [N, 3136, 3]
        
        # N = self.vision1.shape[0]
        # self.vision1 = self.vision1.view(N, 112, 112, 3)              # [N, 112, 112, 3]
        # self.vision1 = self.vision1.permute(0, 3, 1, 2)  # (N, 3, 112, 112)

        # # 2. 归一化，确保输入范围为[0,1]，如果输入范围是[0,255]则先除以255
        # if self.vision1.max() > 1.0:
        #     self.vision1 = self.vision1 / 255.0
        
        # mean = torch.tensor([0.5, 0.5, 0.5], device=self.vision1.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.5, 0.5, 0.5], device=self.vision1.device).view(1, 3, 1, 1)
        # self.vision1 = (self.vision1 - mean) / std
        # # print(self.vision1.shape)
        # # 4. 加载预训练 MobileViT 模型（small）
        # model = timm.create_model("mobilevit_s", pretrained=True)
        # model.eval()
        # model = model.cuda()

        # # 2. 使用 DataLoader 批量处理并提取token
        # image_dataset = ImageOnlyDataset(self.vision1)  # image_tensor shape (N, 112, 112, 3)
        # image_loader = DataLoader(image_dataset, batch_size=40, shuffle=False)
        # # 3. 提取 token
        # image_tokens = []
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # with torch.no_grad():
        #     for batch in image_loader:
        #         batch = batch.to(device)
        #         token = model.forward_features(batch)
        #         token = token.view(token.shape[0], token.shape[1], -1).permute(0, 2, 1)  # (B, D)
        #         image_tokens.append(token.cpu())
        # image_tokens = torch.cat(image_tokens, dim=0)  # shape: (N, D)
        # self.vision1 = image_tokens  # shape: (N, 16, 640)
        # print("vitshape:"+str(self.vision1.shape))
        #################################MobileViT###########################################
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