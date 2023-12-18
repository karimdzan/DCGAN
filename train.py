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



warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(dataloader, gen, disc, fixed_noise, optimizerD, optimizerG, criterion):
    real_label = 1.
    fake_label = 0.
    img_list = []
    iters = 0
    fid = FID()
    for epoch in tqdm(range(200)):
        print(f"Current epoch: {epoch}")
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader)):
            disc.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = disc(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(fake_label)
            output = gen(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            gen.zero_grad()
            label.fill_(real_label)
            output = disc(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            if iters % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, 5, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if (iters % 200 == 0) or ((epoch == 200-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                    fake_scaled = 0.5 * fake + 0.5
                fake_images_loader = DataLoader(fake_scaled, collate_fn=lambda x: {"images": torch.stack(x, dim=0)})
                real_images_loader = DataLoader(data * 0.5 + 0.5, collate_fn=lambda x: {"images": torch.stack(x, dim=0)})
                fake_features = fid.compute_feats(fake_images_loader)
                real_features = fid.compute_feats(real_images_loader)
                fid_score = fid(fake_features, real_features)
                ssim_score = ssim(fake_scaled, data * 0.5 + 0.5, data_range=1.)
                # wandb.log({"fid": fid_score,
                #           "ssim": ssim_score})
                # wandb.log({f"image_{iters}": wandb.Image(vutils.make_grid(fake, padding=2, normalize=True))})
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        if epoch % (50 - 1) == 0:
          torch.save({
                'generator': gen.state_dict(), 
                'discriminator': disc.state_dict(),
                }, 'checkpoint.pth'
            )


def main():
    transform=T.Compose([
                      T.Resize(64),
                      T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    dataset = Base(image_folder='data', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                            shuffle=True, num_workers=2, pin_memory=True)
    
    gen = Generator(**generator_config).to(device)
    disc = Discriminator(**discriminator_config).to(device)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(128, 100, 1, 1, device=device)

    optimizerD = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.99))
    optimizerG = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.99))

    train(dataloader, gen, disc, fixed_noise, optimizerD, optimizerG)

if __name__ == "__main__":
    main()