import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import torchvision
import torch.cuda
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size = 32
image_size = 64
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
nc = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002

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
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 128x16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 256x8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 512x4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),  # 1x1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.disc(input)


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input N x nz x 1 x 1
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 16, kernel_size=4, stride=2, padding=0, bias=False),
            # 1024 x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 512x8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 256x16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 128x32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, padding=1, bias=False),  # 1x64x64
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.ConvTranspose2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
      nn.init.normal_(m.weight.data, 0.0, 0.02)


gen = Generator(nz, nc, ngf).to(device)
disc = Discriminator(nc, ndf).to(device)

"""gen.apply(weights_init)
disc.apply(weights_init)"""

initialize_weights(gen)
initialize_weights(disc)

fixed_noise = torch.randn(32, nz, 1, 1).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

save_dir = "./output"
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(batch_size, nz, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        if batch_idx % 500 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                          Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                torchvision.utils.save_image(img_grid_real, f"{save_dir}/real_epoch{epoch}_batch{batch_idx}.png")
                torchvision.utils.save_image(img_grid_fake, f"{save_dir}/fake_epoch{epoch}_batch{batch_idx}.png")

