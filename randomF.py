# 找出每层随机特征
# 一种是特征均值，一种是二维特征中选取随机特征

import torch
import random

#针对网络G
def findF_G(model,x):
    y = []
    y.append(x)
    for name, layer in model.net._modules.items():
        x = layer(x)
        if isinstance(layer, torch.nn.ConvTranspose2d):
            #print(x.shape)
            y.append(x.mean([2,3]))
    y[0]=y[0].squeeze(2).squeeze(2)
    y[-1]= x
    return y

def findF_Gv2(model,x):
    y = []
    y.append(x)
    for name, layer in model.net._modules.items():
        x = layer(x)
        if isinstance(layer, torch.nn.ConvTranspose2d):
            #print(x.shape)
            flag=random.randint(0,int(x.size(1))-1)
            y.append(x[:,flag])
    y[0]=y[0].squeeze(2).squeeze(2)
    y[-1]= x
    return y

#针对网络D
def findF_D(model,x):
    y = []
    y.append(x)
    for name, layer in model.net._modules.items():
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            #print(x.shape)
            y.append(x.mean([2,3]))
    y[-1]= x.squeeze(2).squeeze(2)
    return y

def findF_Dv2(model,x):
    y = []
    y.append(x)
    for name, layer in model.net._modules.items():
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            #print(x.shape)
            flag=random.randint(0,int(x.size(1))-1)
            y.append(x[:,flag])
    y[-1]=x.squeeze(2).squeeze(2)
    y = y[::-1]
    return y

#test
import networks.D2E_FT as net
G = net.Generator(input_dim = 512, output_channels = 3, image_size = 512, first_hidden_dim_ = 512, last_hidden_dim_= 64)
D = net.Discriminator_SpectrualNorm(input_dim = 512, input_channels = 3, image_size=512, first_hidden_dim_= 64, last_hidden_dim_=512)
x1 = torch.randn(2,3,512,512)
y1 = findF_Dv2(D,x1)
for i in y1:
    print(i.shape)

#x2 = torch.randn(2,512,1,1)
x2 = y1[0].unsqueeze(2).unsqueeze(2)
print(x2.shape)
y2 = findF_Gv2(G,x2)
for i in y2:
    print(i.shape)