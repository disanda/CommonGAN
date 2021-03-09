# coding=utf-8
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class DatasetFromFolder(Dataset):
    def __init__(self,path='',transform=None, channels = 3):
        super().__init__()
        self.channels = channels
        self.path = path
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')] # x.startswith() 
        #imgs_path = os.listdir(path)
        #self.image_filenames = list(filter(lambda x:x.endswith('jpg') or x.endswith('png') ,imgs_path))
    def __getitem__(self, index):
        if self.channels == 1:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L') # 'L'是灰度图, 'RGB'彩色
        elif self.channels ==3:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('RGB')
        else:
            print('error')
        #a = a.resize((self.size, self.size), Image.BICUBIC)
        #a = transforms.ToTensor()(a)
        if self.transform:
            a = self.transform(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

def make_dataset(dataset_name, batch_size,img_size,drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False,img_paths=''):
    if dataset_name == 'mnist' or dataset_name=='fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(), # [0,255] -> [0,1]
            #transforms.Normalize(mean=[0.5], std=[0.5]), # [0,1] -> [1,1] (x*mean+std)
            transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)) # x -> (x,x,x)
        ])
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('data/', transform=transform, download=False)
        else:
            dataset = datasets.FashionMNIST('data/', transform=transform, download=False)
        img_shape = [img_size, img_size, 1]
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10('data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]
    elif dataset_name == 'pose10':
        transform_pose10 = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5]) #黑白应该是不用norm
        ])
        #dataset = DatasetFromFolder(path='/_yucheng/dataSet/pose/pose_set_10',size=img_size)
        #dataset = DatasetFromFolder(path='./data/Pose/pose_set_10',size=img_size)
        path_pose10='./data/Pose/pose_set_10'
        dataset = DatasetFromFolder(path=path_pose10,transform=transform_pose10,channels=1)
        img_shape = [img_size, img_size, 1]
    elif dataset_name == '3dface':
        transform_pose10 = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5]) #黑白应该是不用norm
        ])
        path_pose10='./_yucheng/dataSet/face3d/face3d/'
        dataset = DatasetFromFolder(path=path_pose10,transform=transform_pose10,channels=1)
        img_shape = [img_size, img_size, 3]
    elif dataset_name == 'celeba_64':
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]# [image,height,width]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.Resize(size=(img_size, img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #transforms.ToPILImage()
            ])
        dataset = DatasetFromFolder(path='',size=64)
        img_shape = (img_size, img_size, 3)
    elif dataset_name == 'celeba_HQ':
        transform_a = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(), #Img2Tensor [0,255]->[0,1]
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #取值范围(0,1)->(-1,1)
            ])
        #path_128 = 'F:/dataSet2/CelebAMask-HQ/CelebA-HQ-img' #家主机
        path_a = '/home/disanda/Desktop/dataSet/CelebAMask-HQ/img-30000/CelebA-HQ-img/' #学校个人主机
        #path_128 = '/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img' #云平台
        dataset = DatasetFromFolder(path=path_a,transform=transform_a,channels=3)
        img_shape = (img_size, img_size, 3)
    else:
        raise NotImplementedError
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)
    return data_loader, img_shape

# ==============================================================================
# =                                   debug                                    =
# ==============================================================================

# pose = DatasetFromFolder('/Users/apple/Desktop/AI_code/dataSet/pose_set_1000')

# train_loader = DataLoader(
#      dataset=pose,
#      batch_size=25,#一个batch25张图片,epoch=allData_size/batch_size
#      shuffle=False,
#      #num_workers=0,若是win需要这一行
#      pin_memory=True,#用Nvidia GPU时生效
#      drop_last=True
#  )

# import tqdm
# for x_real in tqdm.tqdm(train_loader, desc='Inner Epoch Loop'):
#     print(type(x_real))

# for i, x in enumerate(train_loader):
#      print(i)
#      print(x.shape)#[n,c,w,h]
#      torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)#这个保存是三通道的,需要改成1通道
