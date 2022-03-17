import math

import torch
import torch.nn as nn

############################# Generator ##################################
class Generator(nn.Module):
    def __init__(self, ngpu, mode=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if mode == 0: # mnist
            # Number of channels in the training images. For color images this is 3
            nc = 1  # 3
            # Size of z latent vector (i.e. size of generator input)
            nz = 100
            # Size of feature maps in generator
            ngf = 64
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf*4, 3, 2, 0, bias=False),
                nn.BatchNorm2d(ngf*4),
                nn.ReLU(True),
                # state size. (ngf*4) x 3 x 3
                nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU(True),
                # state size. (ngf*2) x 8 x 8
                nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 16 x 16
                nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 28 x 28
            )
        elif mode == 1: # CIFAR-10
            # Number of channels in the training images. For color images this is 3
            nc = 3
            # Size of z latent vector (i.e. size of generator input)
            nz = 100
            # Size of feature maps in generator
            ngf = 64
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        return self.main(input)


############################# Discriminator ##################################
class Discriminator(nn.Module):
    def __init__(self, ngpu, mode=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        if mode == 0:  # mnist
            # Number of channels in the training images. For color images this is 3
            nc = 1  # 3
            # Size of feature maps in discriminator
            ndf = 28
            self.main = nn.Sequential(
                # input is (nc) x 28 x 28
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 14 x 14
                nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 7 x 7
                nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf*4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 3 x 3
                nn.Conv2d(ndf*4, 1, 4, 2, 1, bias=False),
                #nn.Sigmoid() # not needed with nn.BCEWithLogitsLoss()
            )
        elif mode == 1:  # CIFAR-10
            # Number of channels in the training images. For color images this is 3
            nc = 3
            # Size of feature maps in discriminator
            ndf = 64
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input)

