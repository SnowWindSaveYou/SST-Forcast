
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self,inplane=64,plane=32,stride=1):
        super(ResNetBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplane,plane,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(plane),
            nn.ReLU(),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane,plane,kernel_size=1,stride=stride),
            nn.BatchNorm2d(plane),
        )
    def forward(self,x):
        y = self.block(x)
        ris = self.downsample(x)
        y+=ris
        return F.relu(y)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1) 
        )
        self.blocks = nn.Sequential(
            ResNetBlock(64,64,1),
            ResNetBlock(64,128,1),
            ResNetBlock(128,256,1),
            ResNetBlock(256,512,1)
        )
        self.fc_layer = nn.Sequential(# 
            nn.Conv2d(512,1,1),
        )
    def forward(self,x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.fc_layer(x)

class PlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np, chosen_idx):
        self.data_set = torch.Tensor(data_np)
        self.label_set = torch.Tensor(label_np).unsqueeze(1)

    def __getitem__(self, index):
        return self.data_set[index], self.label_set[index]

    def __len__(self):
        return self.label_set.size(0)
