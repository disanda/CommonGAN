#逐渐衰减DG,但是D衰减比G慢

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
import networks.network_origin as net
import loss_func
import g_penal


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--dataset', default='pose10')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom'])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.99)
parser.add_argument('--adversarial_loss_mode', default='hinge_v2', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default='none')
parser.add_argument('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])  # !!!
parser.add_argument('--img_size',type=int,default=64)
parser.add_argument('--img_channels', type=int, default=1)# RGB:3 ,L:1
parser.add_argument('--z_dim', type=int, default=64) # 网络随机噪声 z 输入的维度数 即input_dim
parser.add_argument('--device',default='cuda') # 'cpu'
parser.add_argument('--netBias',type=bool,default=False) # 'cpu'
args = parser.parse_args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)

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
#use_gpu = False
device = torch.device(args.device)

# ----------------setup dataset-------------------

if args.dataset in ['cifar10', 'fashion_mnist', 'mnist','pose10']:  # 3: 32x32  4:64:64 5:128 6:256
    data_loader, shape = data.make_dataset(args.dataset, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4




# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = net.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings, biaS = args.netBias).to(device)
D = net.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, biaS = args.netBias).to(device)
with open(output_dir+'/net.txt','w+') as f:
	#if os.path.getsize(output_dir+'/net.txt') == 0: #判断文件是否为空
		print(G,file=f)
		print(D,file=f)


def get_hinge_v2_1():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(0.5- r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(0.5+ f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn


def get_hinge_v2_2():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(0.5- 0.5*r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(0.5+ 0.5*f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -0.5*f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn

def get_hinge_v2_3():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(0.5- 0.25*r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(0.5+ 0.25*f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -0.25*f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn

def get_hinge_v2_4():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(0.5- 0.125*r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(0.5+ 0.125*f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -0.125*f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn

def get_hinge_v2_5():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(0.5- 0.0625*r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(0.5+ 0.0625*f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss
    def g_loss_fn(f_logit):
        f_loss = -0.0625*f_logit.mean()
        return f_loss
    return d_loss_fn, g_loss_fn

def g_loss_fn(f_logit):
    f_loss = -f_logit.mean()
    return f_loss

# adversarial_loss_functions
d_loss_fn_1, g_loss_fn_1 = get_hinge_v2_1()
d_loss_fn_2,g_loss_fn_2 = get_hinge_v2_2()
d_loss_fn_3,g_loss_fn_3 = get_hinge_v2_3()
d_loss_fn_4,g_loss_fn_4 = get_hinge_v2_4()
d_loss_fn_5,g_loss_fn_5 = get_hinge_v2_5()

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(args.beta_1, args.beta_2)) #一阶当前，二阶历史总和
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(args.beta_1, args.beta_2))

#G_optimizer = torch.optim.SGD(G.parameters(), lr=0.0001, momentum=0.9)
#D_optimizer = torch.optim.SGD(D.parameters(), lr=0.0001, momentum=0.9)
#momentum (float, optional): momentum factor (default: 0)
#weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#dampening (float, optional): dampening for momentum (default: 0)
#nesterov (bool, optional): enables Nesterov momentum (default: False)

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

	# try:
	#     ckpt_path = os.path.join(ckpt_dir, 'xxx.ckpt')
	#     ckpt=torch.load(ckpt_path)
	#     ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
	#     D.load_state_dict(ckpt['D'])
	#     G.load_state_dict(ckpt['G'])
	#     D_optimizer.load_state_dict(ckpt['D_optimizer'])
	#     G_optimizer.load_state_dict(ckpt['G_optimizer'])
	# except:
	#     ep, it_d, it_g = 0, 0, 0


	# sample
	sample_dir = os.path.join(output_dir, 'samples_training')
	if not os.path.exists(sample_dir):
		os.mkdir(sample_dir)

	# main loop
	writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))
	z = torch.randn(64, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

	G.train()
	D.train()
	for ep in tqdm.trange(args.epochs+1, desc='Epoch Loop'):
	    it_d, it_g = 0, 0
	    #for x_real,flag in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
	    for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
	        x_real = x_real.to(device)
	        z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)

#--------training D-----------
	        x_fake = G(z)
	        x_real_d_logit = D(x_real)
	        x_fake_d_logit = D(x_fake.detach())

	        #x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
	        
	        #r_loss = torch.max( 0.1+(args.epochs-ep)//args.epochs - x_real_d_logit, torch.zeros_like(x_real_d_logit)).mean()#((args.epochs-ep)//args.epochs) # Round_gp
	        #f_loss = torch.max( 0.1+(args.epochs-ep)//args.epochs + x_fake_d_logit, torch.zeros_like(x_fake_d_logit)).mean()

	        #r_loss = torch.max( ((args.epochs-ep)//args.epochs)*torch.randn(1).to(device) - x_real_d_logit, torch.zeros_like(x_real_d_logit)).mean() #shift_randomD 
	        #f_loss = torch.max( ((args.epochs-ep)//args.epochs)*torch.randn(1).to(device) + x_fake_d_logit, torch.zeros_like(x_fake_d_logit)).mean()

	        #r_loss = torch.max(0.5 - x_real_d_logit, torch.zeros_like(x_real_d_logit)).mean()
	        #f_loss = torch.max(0.5 + x_fake_d_logit, torch.zeros_like(x_fake_d_logit)).mean()

	        if ep < 1000:
	            r_loss = torch.max(0.5 - (args.epochs-ep)/args.epochs*x_real_d_logit, torch.zeros_like(x_real_d_logit)).mean()
	            f_loss = torch.max(0.5 + (args.epochs-ep)/args.epochs*x_fake_d_logit, torch.zeros_like(x_fake_d_logit)).mean()
	        else:
	            r_loss = torch.max(0.4 + (args.epochs-ep)/args.epochs*x_real_d_logit, torch.zeros_like(x_real_d_logit)).mean()
	            f_loss = torch.max(0.4 - (args.epochs-ep)/args.epochs*x_fake_d_logit, torch.zeros_like(x_fake_d_logit)).mean()

	        gp = g_penal.gradient_penalty(functools.partial(D), x_real, x_fake.detach(), gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)
	        D_loss = (r_loss + f_loss) + gp * args.gradient_penalty_weight
	        D_loss = 1/(1+0.002*ep)*D_loss # 渐进式GP!
	        D.zero_grad()
	        D_loss.backward()
	        D_optimizer.step()
	        D_loss_dict={'d_loss': r_loss + f_loss, 'gp': gp}
	        it_d += 1
	        for k, v in D_loss_dict.items():
	            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

#-----------training G-----------
	        x_fake_d_logit_2 = D(x_fake)
	        #G_loss =  (torch.randn(1).to(device)-x_fake_d_logit_2).mean()
	        #G_loss = torch.max( ((args.epochs-ep)//args.epochs)*torch.randn(1).to(device)-x_fake_d_logit_2, torch.zeros_like(x_fake_d_logit_2) ).mean() #* ((args.epochs-ep)//args.epochs) ) #渐进式loss
	        if ep < 1000:
	            G_loss = -0.5*x_fake_d_logit_2.mean()
	        else:
	            G_loss = 0.4*x_fake_d_logit_2.mean()
	        G_loss = 1/(1+0.002*ep)*G_loss # 渐进式GP!
	        G.zero_grad()
	        G_loss.backward()
	        G_optimizer.step()
	        it_g += 1
	        G_loss_dict = {'g_loss': G_loss}
	        for k, v in G_loss_dict.items():
	            writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

#--------------save---------------
	    if ep%10==0:
	        #x_fake = (sample(z)+1)/2
	        with torch.no_grad():
	            z_t = torch.randn(64, args.z_dim, 1, 1).to(device)
	            x_fake = sample(z_t)
	            torchvision.utils.save_image(x_fake,sample_dir+'/ep%d.jpg'%(ep), nrow=8)
	            with open(output_dir+'/loss.txt','a+') as f:
	                    print('Ep:'+str(ep)+'---'+'G_loss:'+str(G_loss)+'------'+'D_loss'+str(D_loss),file=f)
	                    print('------------------------')
	    # save checkpoint
	    if ep%100==0:
	        torch.save(G.state_dict(), ckpt_dir+'/Epoch_G_(%d).pth' % ep)
	        torch.save(D.state_dict(), ckpt_dir+'/Epoch_D_(%d).pth' % ep)
