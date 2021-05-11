import torch
import torch.nn as nn
import torch.nn.functional as F
class MHSA(nn.Module):
  def __init__(self, n_dims, width, height):
    super(MHSA, self).__init__()

    self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
    self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
    self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
    self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
    self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    n_batch, C, width, height = x.size()
    q = self.query(x).view(n_batch, C, -1)
    k = self.key(x).view(n_batch, C, -1)
    v = self.value(x).view(n_batch, C, -1)
    content_content = torch.bmm(q.permute(0, 2, 1), k)

    content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
    content_position = torch.matmul(content_position, q)

    energy = content_content + content_position
    attention = self.softmax(energy)

    out = torch.bmm(v, attention.permute(0, 2, 1))
    out = out.view(n_batch, C, width, height)

    return out

class BottleBlock(nn.Module):
  def __init__(self,inplane,plane,width,height):
    super(BottleBlock, self).__init__()
    self.block = nn.Sequential(
            nn.Conv2d(inplane,plane,kernel_size=1,stride=1,bias=True),
            MHSA(plane,width,height),
            # nn.Conv2d(plane,plane*4,kernel_size=1,stride=1,bias=True),
            nn.Conv2d(plane,plane,kernel_size=1,stride=1,bias=True),
        )
    self.downsample = nn.Sequential(
            # nn.Conv2d(inplane,plane*4,kernel_size=1,bias=True),
            # nn.BatchNorm2d(plane*4),
            nn.Conv2d(inplane,plane,kernel_size=1,bias=True),
            nn.BatchNorm2d(plane),
        )

  def forward(self, x):
    y = self.block(x)
    x = self.downsample(x)
    x += y
    x = F.relu(x)
    return x

class BotNet(nn.Module):
    def __init__(self,dim ,width, height):
        super(BotNet,self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(dim,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1) 
        )
        self.blocks = nn.Sequential(
            BottleBlock(64,64,width,height),
            BottleBlock(256,64,width,height),
            BottleBlock(256,64,width,height),
        )
        self.out_layer = nn.Sequential(# 
            nn.Conv2d(1024,3,1,1),
            nn.ReLU(),
            nn.Conv2d(3,1,1),
        )
    def forward(self,x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.out_layer(x)

class PlotDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np):
        self.data_set = torch.Tensor(data_np)
        self.label_set = torch.Tensor(label_np).unsqueeze(1)

    def __getitem__(self, index):
        return self.data_set[index], self.label_set[index]

    def __len__(self):
        return self.label_set.size(0)