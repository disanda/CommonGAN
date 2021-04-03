import functools
import numpy as np
import tensorboardX
import torch
import tqdm
import argparse
import os
import yaml
import torchvision
import data
import networks.D2E as net
import loss_func
import g_penal
from torchsummary import summary
import matplotlib.pyplot as plt

# command line
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default='none')
parser.add_argument('--img_size',type=int, default=256)
parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
parser.add_argument('--dataset', default='mnist')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
parser.add_argument('--z_dim', type=int, default=256)
parser.add_argument('--Gscale', type=int, default=8) # scale：网络隐藏层维度数,默认为 image_size//8 * image_size 
parser.add_argument('--Dscale', type=int, default=1) # dscale：网络参数和G的比例，默认为1，即1比1
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = net.Generator(input_dim=args.z_dim, output_channels = args.img_channels, image_size=args.img_size, scale=args.Gscale).to(device)
D = net.Discriminator_SpectrualNorm(input_dim=args.z_dim, input_channels = args.img_channels, image_size=args.img_size, Gscale=args.Gscale, Dscale=args.Dscale).to(device)
summary(G,(256,1,1))
summary(D,(3,256,256))
G.load_state_dict(torch.load('./pre-model/cat/cat256_Gs_dict.pth'))


def viz(module, input):
    feature_num=10
    x = input[0][0]
    min_num = np.minimum(feature_num, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, feature_num, i+1)
        plt.imshow(x[i].cpu())
    plt.show()

def main():
    for name,m in G.named_modules():
        # print('------------')
        # print(name)
        # print(m)
        if isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_pre_hook(viz)
    z = torch.randn(6, args.z_dim, 1, 1).to(device)
    with torch.no_grad():
        img = G(z)

if __name__ == '__main__':
    main()

