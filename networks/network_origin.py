import functools
import torch
from torch import nn

class Identity(torch.nn.Module):
    def __init__(self, *args, **keyword_args):
        super().__init__()
    def forward(self, x):
        return x

def _get_norm_layer_2d(norm):
    if norm == 'none':
        return Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError

class ConvGenerator(nn.Module):
    def __init__(self,
                 input_dim=128,
                 output_channels=3,
                 dim=64,
                 n_upsamplings=4,
                 norm='batch_norm',
                 biaS = False
                ):
        super().__init__()
        Norm = _get_norm_layer_2d(norm)
        self.bias = biaS
        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1,biass=False):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=biass),
                Norm(out_dim),
                nn.ReLU()
            )
        layers = []
        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0, biass = self.bias))
        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = d//2
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1,biass=self.bias))
        layers.append(nn.ConvTranspose2d(d, output_channels, kernel_size=4, stride=2, padding=1, bias=self.bias))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        x = self.net(z)
        return x

class ConvDiscriminator(nn.Module):
    def __init__(self,
                 input_channels=3,
                 dim=64,
                 n_downsamplings=4,
                 norm='batch_norm',
                 biaS = False
                 ):
        super().__init__()
        Norm = _get_norm_layer_2d(norm)
        self.bias = biaS
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, biass=False):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=biass),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        layers = []
        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1, bias=self.bias))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = d*2 # 1 > 2 > 4 > 8  > 16 (64*64) > 32 > 64 >128 > 256 (1024*1024) 
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1, biass=self.bias))
        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0, bias=self.bias))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        y = self.net(x)
        return y # [1,1,1,1] 
