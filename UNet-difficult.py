from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM


import time
import psutil
import os
import subprocess

#reform the picture
img_size = 64
batch_size = 16
train_path = "./front"
train_path2 = "./Abstract_gallery_2"

def load_trans_dataset():
    data_trans = [
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t:(t*2)-1)
    ]

    tran = transforms.Compose(data_trans)
    # load first dataset
    train = torchvision.datasets.ImageFolder(root=train_path,
                                             transform=tran)
    # load second dataset
    train2 = torchvision.datasets.ImageFolder(root=train_path2,
                                              transform=tran)
    # this will combin the datasets together
    return torch.utils.data.ConcatDataset([train,train2])

def train_cifar10(
    n_epoch: int = 1001, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    # if load_pth is not None:
    #     ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    ddpm.to(device)

    # tf = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    #
    # dataset = CIFAR10(
    #     "./data",
    #     train=True,
    #     download=True,
    #     transform=tf,
    # )
    #
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    data = load_trans_dataset()
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    optim = torch.optim.Adam(ddpm.parameters(), lr=0.001)

    # epochs = 5 # Try more!
    epochs = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 400, 500, 600, 700, 800, 900, 1000]  # Try more!
    # epochs = [1] # Try more!

    timelsit = []
    gpulist = []
    # for i in epochs:
    time_start = time.time()

    # m_state_dict = torch.load(f'train_cifar10weightPT/1000_train_cifar10.pt')
    # ddpm.load_state_dict(m_state_dict)

    for epoch in range(n_epoch):
        print(f"Epoch {epoch} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            # xh = ddpm.sample(1, (3, 64, 64), device)
            # xset = torch.cat([xh], dim=0)
            # grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=1)
            # save_image(grid, f"./contents/ddpm_sample_cifar{epoch+1000}.png")
            if epoch % 100 == 0:
                print("start sampling")
                for i in range(10):
                    xh = ddpm.sample(1, (3, 64, 64), device)
                    xset = torch.cat([xh], dim=0)
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=1)
                    save_image(grid, f"./train_cifar10PT_out/testOutput{epoch}/testOutput{epoch}/ddpm_sample_cifar{i}.png")
                    print(f"save {i} image")

            if epoch in epochs:
                # # save trained data from the model
                # torch.save(ddpm.state_dict(), f'./train_cifar10weightPT/{epoch}_train_cifar10.pt')

                time_end = time.time()
                time_sum = time_end - time_start
                print(time_sum)
                timelist.append(time_sum)

                # using nvidia-smi command get GPU memory
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE)
                # decode the output
                output = result.stdout.decode('utf-8')

                # get the memory used
                for line in output.strip().split('\n'):
                    total, used = line.split(', ')
                    gpulist.append({"total_memory_MB": int(total), "used_memory_MB": int(used)})

            print(timelist)
            print(gpulist)


if __name__ == "__main__":
    train_cifar10()
