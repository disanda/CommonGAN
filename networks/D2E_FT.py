#v1这一版调整了参数，最大参数为512，并调大了原先DCGAN中参数较小的层

#v2加了残差，用1*1卷积融合了特征层保证残差通道对等


import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
from torch.nn import functional as F

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)

class Generator(nn.Module):
    def __init__(self, input_dim=512, output_channels=3, image_size=512, first_hidden_dim_ = 512, last_hidden_dim_ =64, another_times=0):
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 - another_times # 减去前两次 1->2->4， 及最后一次， 方便中间写循环
        first_hidden_dim = first_hidden_dim_ # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 
        layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag)) # 1*1 input -> 4*4
        #layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=2,padding=1,bias=bias_flag)) # 4*4 input -> 8*8
        #layers.append(nn.BatchNorm2d(first_hidden_dim))
        layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        hidden_dim = first_hidden_dim
        while up_times>0:
            if up_times < (math.log(first_hidden_dim_,2) - math.log(last_hidden_dim_,2)+1):
                layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
                #layers.append(nn.BatchNorm2d(hidden_dim//2))
                layers.append(nn.InstanceNorm2d(hidden_dim//2, affine=False, eps=1e-8))
                layers.append(nn.ReLU())
                hidden_dim = hidden_dim // 2
            else:
                layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
                #layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
                layers.append(nn.ReLU())
            up_times = up_times - 1

        # 3:end 
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_dim=512, input_channels=3, image_size=512, first_hidden_dim_=64, last_hidden_dim_=512, another_times=0): #新版的Dscale是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3 - another_times
        first_hidden_dim = first_hidden_dim_
        bias_flag = False

        # 1:
        #layers.append(spectral_norm(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:  
            if hidden_dim < last_hidden_dim_:
                #layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
                layers.append(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag))
                layers.append(nn.InstanceNorm2d(hidden_dim*2, affine=False, eps=1e-8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                hidden_dim = hidden_dim * 2
            else:
                #layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag))
                layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            up_times = up_times - 1
        # 3:
        layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=1, padding=0)) # 4*4 > 1*1
        #layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1)) # 8*8 > 4*4

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        #y = y.mean()
        return y # [1,1,1,1]


def upscale2d(x, factor=2, conv1=False):
    #    return F.upsample(x, scale_factor=factor, mode='bilinear', align_corners=True)
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    #if conv1:
    return x

def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class Generator_v2(nn.Module):
    def __init__(self, input_dim=512, output_channels=3, image_size=512, first_hidden_dim_ = 512, last_hidden_dim_ =64, another_times=0):
        super().__init__()
        up_times = math.log(image_size,2)- 3 - another_times # 减去前两次 1->2->4， 及最后一次， 方便中间写循环
        first_hidden_dim = first_hidden_dim_ # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 
        layers = []
        layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag)) # 1*1 input -> 4*4
        #layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=2,padding=1,bias=bias_flag)) # 4*4 input -> 8*8
        #layers.append(nn.BatchNorm2d(first_hidden_dim))
        layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.ReLU())
        setattr(self, "layer%d" % (1), nn.Sequential(*layers))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        hidden_dim = first_hidden_dim
        i=2
        while up_times>0:
            if up_times < (math.log(first_hidden_dim_,2) - math.log(last_hidden_dim_,2)+1):
                layers = []
                #layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
                layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
                #layers.append(nn.BatchNorm2d(hidden_dim//2))
                layers.append(nn.InstanceNorm2d(hidden_dim//2, affine=False, eps=1e-8))
                layers.append(nn.ReLU())
                setattr(self, "layer%d" % (i), nn.Sequential(*layers))
                layers = []
                layers.append(nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1))
                layers.append(nn.ReLU())
                setattr(self, "conv1_%d" % (i), nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1))
                hidden_dim = hidden_dim // 2
            else:
                layers = []
                layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
                #layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
                layers.append(nn.ReLU())
                setattr(self, "layer%d" % (i), nn.Sequential(*layers))
            i = i +1
            up_times = up_times - 1

        # 3:end 
        layers = []
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())
        setattr(self, "layer%d" % (i), nn.Sequential(*layers))


    def forward(self, z):
        x = upscale2d(z,factor=4)
        x = getattr(self, "layer%d" % (1))(z) + 0.9*x
        for i in range(int(math.log(512,2))-3):
            if hasattr(self,'conv1_%d' % (i+2)):
                x = 0.9*getattr(self, "layer%d" % (i+2))(x) + 0.1*upscale2d(x)
                x = getattr(self,'conv1_%d'%(i+2))(x)
            else:
                x = 0.9*getattr(self, "layer%d" % (i+2))(x) + 0.1*upscale2d(x)
        x = getattr(self, "layer%d" % (int(math.log(512,2))-1))(x)
        return x

class Discriminator_SpectrualNorm_v2(nn.Module):
    def __init__(self, input_dim=512, input_channels=3, image_size=512, first_hidden_dim_=64, last_hidden_dim_=512, another_times=0): #新版的Dscale是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3 - another_times
        first_hidden_dim = first_hidden_dim_
        bias_flag = False

        # 1:
        layers = []
        layers.append(spectral_norm(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        #layers.append(nn.Conv2d(input_channels, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag))
        #layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        setattr(self, "layer%d" % (1), nn.Sequential(*layers))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        i=2
        while up_times>0:  
            if hidden_dim < last_hidden_dim_:
                layers = []
                layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
                #layers.append(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1, bias=bias_flag))
                #layers.append(nn.InstanceNorm2d(hidden_dim*2, affine=False, eps=1e-8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                setattr(self, "layer%d" % (i), nn.Sequential(*layers))
                layers = []
                layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1)))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                setattr(self, "conv1_%d" % (i), nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1))
                hidden_dim = hidden_dim * 2
            else:
                layers = [] 
                layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
                #layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag))
                #layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                setattr(self, "layer%d" % (i), nn.Sequential(*layers))
            up_times = up_times - 1
            i = i + 1
        
        # 3:
        layers = [] 
        layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=1, padding=0)) # 4*4 > 1*1
        #layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1)) # 8*8 > 4*4
        setattr(self, "layer%d" % (i), nn.Sequential(*layers))

    def forward(self, x):
        z = getattr(self, "layer%d" % (1))(x)
        for i in range(int(math.log(512,2))-3):
            if hasattr(self,'conv1_%d' % (i+2)):
                z = 0.9*getattr(self, "layer%d" % (i+2))(z) + 0.1*downscale2d(z)
                z = getattr(self,'conv1_%d'%(i+2))(z)
            else:
                z = 0.9*getattr(self, "layer%d" % (i+2))(z) + 0.1*downscale2d(z)
        z_ = downscale2d(z,factor=4)
        z = getattr(self, "layer%d" % (int(math.log(512,2))-1))(z) + 0.9*z_
        return z



