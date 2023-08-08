import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 32
image_size = 64
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
nc = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),  # 0 ~ 1
    transforms.Normalize(mean=(0.5), std=(0.5))  # -1 ~ 1
])

# dataset
dataset = datasets.MNIST(root="data/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input N x 1 x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False), # 64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), #128x16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 256x8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # 512x4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False), # 1x1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.disc(input)


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input N x nz x 1 x 1
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 16, kernel_size=4, stride=2, padding=0, bias=False), # 1024 x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False), # 512x8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), #256x16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 128x32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, padding=1, bias=False), # 1x64x64
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)



