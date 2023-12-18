import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, stride, padding):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
              in_channels, 
              hidden_dim * 8, 
              kernel_size, 
              1, 
              0, 
              bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(
              hidden_dim * 8, 
              hidden_dim * 16, 
              kernel_size, 
              stride, 
              padding, 
              bias=False),
            nn.BatchNorm2d(hidden_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d( 
              hidden_dim * 16, 
              hidden_dim * 8, 
              kernel_size, 
              stride, 
              padding, 
              bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d( 
              hidden_dim * 8, 
              hidden_dim * 4, 
              kernel_size, 
              stride, 
              padding, 
              bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(
              hidden_dim * 4, 
              out_channels, 
              kernel_size, 
              stride, 
              padding, 
              bias=False),
            nn.Tanh()
        )
        self.main.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, stride, padding):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, out_channels, kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.main.apply(weights_init)

    def forward(self, input):
        return self.main(input)