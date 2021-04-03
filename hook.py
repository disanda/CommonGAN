import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import cv2
import numpy as np

def viz(module, input):
    print('??????')
    print(input[0].shape)
    print('??????')
    x = input[0][0]
    min_num = np.minimum(6, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 6, i+1)
        plt.imshow(x[i].cpu())
    plt.show()

def main():
    t = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0]),
        ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=True).to(device)
    for name,m in model.named_modules():
        print(name)
        print('++++++++++++++++')
        print(m)
        print('----------------')
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)
    img = cv2.imread('./Unknown.jpg')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)
    summary(model, (3, 64, 64))


if __name__ == '__main__':
    main()