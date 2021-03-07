import torch 
import torchvision 
import networks as net
import encoders
import os
import torch.nn as nn
import torch.optim as optim
import tqdm
import data

# select the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

if __name__ == '__main__':
#----------------------配置预训练模型------------------
    G = net.ConvGenerator(128, 1, n_upsamplings=4).to(device)# in: [-1,128], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
    G.load_state_dict(torch.load('./pre_trained/hinge-gp_G_(19).pth',map_location=device)) #shadow的效果要好一些 

    # #netD1 = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
    # #netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))
    E = encoders.encoder_v3()
    # toggle_grad(netD1,False)
    # toggle_grad(netD2,False)
    # paraDict = dict(netD1.named_parameters()) # pre_model weight dict
    # for i,j in netD2.named_parameters():
    #     if i in paraDict.keys():
    #         w = paraDict[i]
    #         j.copy_(w)
    # toggle_grad(netD2,True)
    # del netD1
#print(netG)
#print(netD1)

#test
    # z = torch.randn(8,128,1,1)
    # x = (G(z)+1)/2
    # print(x.shape)#(8,1,64,64)
    # torchvision.utils.save_image(x,'./test.jpg', nrow=8)

    
    #sample_dir = os.path.join('./gan_samples_rc_v1') # v1
    #sample_dir = os.path.join('./gan_samples_rc_v2') # v2
    sample_dir = os.path.join('./gan_samples_rc_v3') # v3
    #sample_dir = os.path.join('./gan_samples_rc_v4') # v4
    #sample_dir = os.path.join('./gan_samples_rc_v5') # v5
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)


    data_loader, shape = data.make_dataset('fashion_mnist', batch_size=8, img_size=64,pin_memory=True,shuffle=False)
    n_G_upsamplings = n_D_downsamplings = 4


#------------train-------------
    criterion = nn.MSELoss()
    optimizerE = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #z = torch.randn(8, 128, 1, 1).to(device)  # a fixed noise for sampling
    for ep in tqdm.trange(10, desc='Epoch Loop'):
        it_d, it_g = 0, 0
        for x_real,flag in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
            x_real = x_real.to(device)
            z_ = E(x_real)
            x_fake = G(z_)
            err = criterion(x_fake,x_real)
            E.zero_grad()
            err.backward()
            err_item = err.mean().item()
            optimizerE.step()
            it_g += 1
        # sample
            if it_g % 1 == 0:
                print('--------')
                print(err_item)
                img = torch.cat((x_real,x_fake))
                img = (img+1)/2
                torchvision.utils.save_image(img,sample_dir+'/ep%d_%d.jpg'%(ep,it_g), nrow=8)
    # save checkpoint
    if (ep+1)%5==0:
        torch.save(E.state_dict(), sample_dir+'/Epoch_E_(%d).pth' % ep)
