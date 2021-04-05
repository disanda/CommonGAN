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
import networks.D2E_2Conv as net
import loss_func
import g_penal
from torchsummary import summary


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default='none')
parser.add_argument('--img_size',type=int, default=512)
parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
parser.add_argument('--dataset', default='Celeba_HQ')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom','Celeba_HQ'])
parser.add_argument('--z_dim', type=int, default=512)
parser.add_argument('--Gscale', type=int, default=4) # scale：网络隐藏层维度数,默认为 image_size//8 * image_size 
parser.add_argument('--Dscale', type=int, default=1) 
args = parser.parse_args()
another_times_=1 #输入是4*4时需要

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)

args.experiment_name += '_Gs%d_Ds%d_Zdim%d_imgSize%d_batch_size%d_4*4_onemoreConv' % (args.Gscale, args.Dscale, args.z_dim, args.img_size,args.batch_size)

output_dir = os.path.join('output', args.experiment_name)

if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# save settings

with open(os.path.join(output_dir, 'settings.yml'), "w", encoding="utf-8") as f:
    yaml.dump(args, f)


# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# dataset
data_loader, shape = data.make_dataset(args.dataset, args.batch_size, args.img_size,pin_memory=use_gpu)
#n_G_upsamplings = n_D_downsamplings = 5 # 3: 32x32  4:64:64 5:128 6:256
print('data-size:    '+str(shape))

# ==============================================================================
# =                                   model                                    =
# ==============================================================================
G = net.Generator(input_dim=args.z_dim, output_channels = args.img_channels, image_size=args.img_size, scale=args.Gscale, another_times=another_times_).to(device)
D = net.Discriminator_SpectrualNorm(input_dim=args.z_dim, input_channels = args.img_channels, image_size=args.img_size, Gscale=args.Gscale, Dscale=args.Dscale, another_times=another_times_).to(device)
#G.load_state_dict(torch.load('./pre-model/G_in256_G8.pth',map_location=device)) #shadow的效果要好一些 
#D.load_state_dict(torch.load('./pre-model/D_in256_D4.pth',map_location=device))
summary(G,(512,4,4))
summary(D,(3,512,512))
x,y = net.get_parameter_number(G),net.get_parameter_number(D)
x_GB, y_GB = net.get_para_GByte(x),net.get_para_GByte(y)


with open(output_dir+'/net.txt','w+') as f:
    #if os.path.getsize(output_dir+'/net.txt') == 0: #判断文件是否为空
        print(G,file=f)
        print(x,file=f)
        print(x_GB,file=f)
        print('-------------------',file=f)
        print(D,file=f)
        print(y,file=f)
        print(y_GB,file=f)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = loss_func.get_adversarial_losses_fn(args.adversarial_loss_mode)


# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
#decayG = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=1)
#decayD = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=1)


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

if __name__ == '__main__':
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # sample
    sample_dir = os.path.join(output_dir, 'samples_training')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # main loop
    writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))

    G.train()
    D.train()
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        it_d, it_g = 0, 0
        for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
            if args.dataset == ('mnist' or 'fashion_mnist'):
                x_real = x_real[0].to(device) # x_real[1]是标签
            else:
                x_real = x_real.to(device)
            z = torch.randn(args.batch_size, args.z_dim, 4, 4).to(device)
            #z = torch.randn(args.batch_size, args.z_dim, 4, 4).to(device) #PGGAN
#--------training D-----------
            x_fake = G(z)
            #print(x_real.shape)
            x_real_d_logit = D(x_real)
            x_fake_d_logit = D(x_fake.detach())

            x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)

            #gp = g_penal.gradient_penalty(functools.partial(D), x_real, x_fake.detach(), gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)
            gp = torch.tensor(0.0)
            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight
            #D_loss = 1/(1+0.005*ep)*D_loss # 渐进式GP!

            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            #decayD.step()

            D_loss_dict={'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

            it_d += 1
            for k, v in D_loss_dict.items():
                writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

#-----------training G-----------
            x_fake_d_logit_2 = D(x_fake)
            G_loss = g_loss_fn(x_fake_d_logit_2) #渐进式loss
            #G_loss = 1/(1+ep*0.01)*g_loss_fn(x_fake_d_logit) #渐进式loss
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            #decayG.step()

            it_g += 1
            G_loss_dict = {'g_loss': G_loss}
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

#--------------save---------------
            if (it_g)%100==0:
                with torch.no_grad():
                    torchvision.utils.save_image(x_fake,sample_dir+'/ep%d_it%d.jpg'%(ep,it_g), nrow=args.batch_size//4)
                    with open(output_dir+'/loss.txt','a+') as f:
                        print('G_loss:'+str(G_loss)+'------'+'D_loss'+str(D_loss),file=f)
                        print('------------------------')
        # save checkpoint
        if (ep+1)%10==0:   
            #torch.save(G.state_dict(), ckpt_dir+'/Epoch_G.pth') #保存每次需要覆盖
            #torch.save(D.state_dict(), ckpt_dir+'/Epoch_D.pth')
            torch.save(G.state_dict(), ckpt_dir+'/Epoch_G_%d.pth' % ep)
            torch.save(D.state_dict(), ckpt_dir+'/Epoch_D_%d.pth' % ep)
