# ----------自己写的一套DCGAN网络，可以通过图像分辨率调整网络参数，包括输入维度，中间维度.--------
#在 512 * 512 分辨率上, 4*4输入测试: 
#在特征较少的block上再加一层conv, torch.nn.Conv2d(3,5,3,1,1)
#其他改进: a.加残差 b.加AE训练

import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class Generator(nn.Module):
    def __init__(self, input_dim=128, output_channels=3, image_size=128, scale=16, another_times=0):
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 - another_times # 减去前两次 1->2->4， 及最后一次， 方便中间写循环
        first_hidden_dim = image_size*scale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 1x1 -> 4x4
        layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=2,padding=1,bias=bias_flag))
        layers.append(nn.BatchNorm2d(first_hidden_dim))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        hidden_dim = first_hidden_dim
        while up_times>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.BatchNorm2d(hidden_dim//2))
            layers.append(nn.ReLU())
            up_times = up_times - 1
            hidden_dim = hidden_dim // 2

        # 3:end
        layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=3, stride=1, padding=1 ,bias=bias_flag))
        layers.append(nn.BatchNorm2d(hidden_dim//2))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(hidden_dim//2,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_dim=128, input_channels=3, image_size=128, Gscale=16, Dscale=1, another_times=0): #新版的Dscale是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3 - another_times
        first_hidden_dim = (input_dim * Gscale// 2**int(up_times)) // Dscale # 默认为input_dim 
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(spectral_norm(nn.Conv2d(first_hidden_dim, first_hidden_dim//2, kernel_size=3, stride=1, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:  
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        # 3: 4*4 > 1*1
        layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1))

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        #y = y.mean()
        return y # [1,1,1,1]


