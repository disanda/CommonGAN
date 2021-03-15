# ----------自己写的一套DCGAN网络，可以通过图像分辨率调整网络参数，包括输入维度，中间维度.--------

# 1. 以及不同分辨率的对应不同上采样数(即网络的层数)，默认第一次上采样，像素从 1->4。 之后每一次上采样，像素增加一倍(这里应该是长宽都增加一倍).
# 2. input_dim 对应G的《输入维度》，image_size表示《生成图片对应的像素》, first_hidden_dim对应《网络中间层维度》(中间层起点的维度)
# 3. scale是input_dim放大的倍数，用于决定中间隐藏层起始时的size, 其和input_dim共同决定网络参数的规模.
# 4. 如果希望G的输入维度 和 D的输出维度对称, 则 first_hidden_dim = input_dim * scale

# 测试网络规模:
# import networks.network_1 as net
# G = net.Generator(input_dim=32, image_size=256, scale=32)
# D = net.Discriminator_SpectrualNorm(input_dim=32, image_size=256, scale=16)
# x,y = net.get_parameter_number(G),net.get_parameter_number(D)
# x_G, y_G = net.get_para_GByte(G),net.get_para_GByte(D)

#D2E , 在DCGAN下，比较E的参数和G的参数对性能的影响
#D2E_PG , 在PGGAN(EQL和PixelNormLyer)下比较s

import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
import torch.nn.functional as F

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class PixelNormLayer(nn.Module):
    def __init__(self, inchannel=0, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None
    def forward(self, input):
        out = F.conv2d(input,self.weight * self.scale,bias=self.bias,stride=self.stride,padding=self.padding)
        return out

class Generator(nn.Module):
    def __init__(self, up_times=5, bias_flag = False): # up_times=5 就是256*256
        super().__init__()
        layers = []
        bias_flag = False

        # 1: 1x1 -> 4x4
        layers.append(EqualConv2d(512, 512, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        #layers.append(PixelNormLayer(512))
        layers.append(nn.ReLU())

        layers.append(EqualConv2d(512, 512, kernel_size=3,stride=1,padding=1,bias=bias_flag))
        #layers.append(PixelNormLayer(512))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32 -> 64 -> 128
        while up_times>0:
            layers.append(EqualConv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=bias_flag))
            #layers.append(PixelNormLayer(512))
            layers.append(nn.ReLU())
            layers.append(EqualConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag))
            #layers.append(PixelNormLayer(512))
            layers.append(nn.ReLU())
            up_times = up_times - 1


        # 3:end 
        layers.append(EqualConv2d(512,256, kernel_size=4,stride=2, padding=1, bias=bias_flag))
        #PixelNormLayer()
        layers.append(nn.ReLU())


        layers.append(EqualConv2d(256,3,kernel_size=3, padding=1, bias=bias_flag))
        #PixelNormLayer()
        layers.append(nn.Tanh())


        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, up_times=5, bias_flag = False):
        super().__init__()
        layers=[]

        # 1:
        layers.append(EqualConv2d(3, 256, kernel_size=3, padding=1, bias=bias_flag))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(EqualConv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        while up_times>0:
            layers.append(EqualConv2d(512, 512, kernel_size=3, padding=1, bias=bias_flag))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(EqualConv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=bias_flag))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        layers.append(EqualConv2d(512, 512, kernel_size=3, padding=1, bias=bias_flag))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(EqualConv2d(512, 512, kernel_size=4, stride=1, padding=0))
        #layers.append(nn.Sigmoid())

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        y = y.mean()
        return y # [1,1,1,1]


