import argparse
import os
import random
import warnings
import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from data import Base
from model import Generator, Discriminator
from piq import FID, ssim

from torch.utils.data import DataLoader
from config import generator_config, discriminator_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    generator = Generator(**generator_config).to(device)
    generator.load_state_dict(torch.load('checkpoint.pth')['generator'])
    random_noise = torch.randn(64, 100, 1, 1, device=device)
    generator.eval()
    with torch.no_grad():
        generated_images = generator(random_noise).detach().cpu()

    image_grid = vutils.make_grid(generated_images, padding=2, normalize=True)
    vutils.save_image(image_grid, 'eval.png')

if __name__ == "__main__":
    main()