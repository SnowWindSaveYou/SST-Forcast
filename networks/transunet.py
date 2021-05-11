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

class MHSARsd(nn.Module):
    def __init__(self,  in_channels,out_channels, width, height):
        super(MHSARsd,self).__init__()
        self.down = in_channels!=out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,128,1,1),
            MHSA(128, width, height),
            nn.Conv2d(128,out_channels,1,1),
            nn.BatchNorm2d(out_channels)
        )
        if self.down:
          self.downsampler = nn.Sequential(
              nn.Conv2d(in_channels,out_channels,1,1),
              nn.BatchNorm2d(out_channels)
          )
    def forward(self, x):
        y = self.block(x)
        if self.down:
          x = self.downsampler(x)
        y+=x
        return F.relu(y)

# Unet Conponents
class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown,self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            UnetBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.block(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = UnetBlock(in_channels, out_channels)

    def forward(self, x,g):
        x_up = self.up(x)
        y = torch.cat([g,x_up],1)
        return self.conv(y)

class UnetBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetTEST,self).__init__()

        layers = [64,128,256,512]
        self.input_layer = UnetBlock(3,layers[0])
        self.down1 = UnetDown(layers[0],layers[1])#12,36
        self.down2 = UnetDown(layers[1],layers[2])#6,18
        self.down3 = UnetDown(layers[2],layers[3])#3,9

        self.up1 = UnetUp(layers[3]+layers[2],layers[2])
        self.up2 = UnetUp(layers[2]+layers[1],layers[1])
        self.up3 = UnetUp(layers[1]+layers[0],layers[0])
        self.out = nn.Sequential(
            UnetBlock(layers[0],layers[0]),
            nn.Conv2d(layers[0],out_channels,1,1)
        )

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        y1 = self.up1(x4,x3)
        y2 = self.up2(y1,x2)
        y3 = self.up3(y2,x1)
        return self.out(y3)

class UnetTran(nn.Module):
  def __init__(self, in_channels, out_channels):
      super(UnetTran,self).__init__()

      layers = [64,128,256,512]
      self.input_layer = UnetBlock(3,layers[0])
      self.down1 = UnetDown(layers[0],layers[1])#12,36
      self.down2 = UnetDown(layers[1],layers[2])#6,18
      self.down3 = UnetDown(layers[2],layers[3])#3,9

      self.up1 = UnetUp(layers[3]+layers[2],layers[2])
      self.up2 = UnetUp(layers[2]+layers[1],layers[1])
      self.up3 = UnetUp(layers[1]+layers[0],layers[0])
      self.out = nn.Sequential(
            UnetBlock(layers[0],layers[0]),
            nn.Conv2d(layers[0],out_channels,1,1)
        )

      ## 
      self.tran1 = MHSARsd(layers[3],layers[3],3,9)
      self.tran2 = MHSARsd(layers[3],layers[3],3,9)
      self.tran3 = MHSARsd(layers[3],layers[3],3,9)
      self.tran4 = MHSARsd(layers[3],layers[3],3,9)
      ##

  def forward(self, x):
      x1 = self.input_layer(x)
      x2 = self.down1(x1)
      x3 = self.down2(x2)
      x4 = self.down3(x3)

      ##
      t1 = self.tran1(x4)
      t2 = self.tran2(t1)
      t3 = self.tran2(t2)
      t4 = self.tran2(t3)
      ##

      y1 = self.up1(t4,x3)
      y2 = self.up2(y1,x2)
      y3 = self.up3(y2,x1)
      return self.out(y3)