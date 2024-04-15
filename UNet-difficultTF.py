from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import pathlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Model
from PIL import Image
import psutil
import os
import subprocess

from mindiffusion.unettf import NaiveUnet
from mindiffusion.ddpmtf import DDPM


def saveImage(images, filename, nrow=4):
    images = (images + 1) * 127.5
    images = tf.clip_by_value(images, 0, 255)
    images = tf.cast(images, tf.uint8)
    # change to numpy can use by plt
    images = images.numpy()
    # have nrow * nrow form of images
    fig, axes = plt.subplots(nrows=nrow, ncols=nrow, figsize=(nrow, nrow), dpi=64)
    # make it can use for loop
    if nrow == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # go trough all the pic
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    # remove all the space between pictures
    plt.tight_layout(pad=0)
    plt.savefig(filename)
    plt.close()


train_path = pathlib.Path('./front')
# reform the picture
img_size = 64
batch_size = 16
BUFFER_SIZE = 1000

def load_and_transform_dataset(train_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(img_size, img_size))

    # Define a function to normalize images to [-1, 1]
    def normalize(image,x):
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image
    # apply normalization
    train_dataset = train_dataset.map(normalize, tf.data.AUTOTUNE)
    # apply shuffle
    train_dataset = train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)

    return train_dataset

def train_cifar10(
    n_epoch: int = 1001, device: str = "/GPU:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    # load data
    dataset = load_and_transform_dataset(train_path)
    dataset = dataset.apply(tf.data.experimental.unbatch())
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    #optimizet
    optim = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # epochs = 5 # Try more!
    epochs = [1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 400, 500, 600, 700, 800, 900, 1000]  # Try more!


    timelsit = []
    gpulist = []
    # for i in epochs:
    time_start = time.time()
    # saving model
    # checkpoint = tf.train.Checkpoint(optimizer=optim, model=ddpm)
    # restore = checkpoint.restore(f'train_cifar10weightTF/1_superminddpm_TF/1_train_cifar10.ckpt-1')
    for epoch in range(n_epoch):
        # if Path("./ddpm_mnist.ckpt").exists():
        #     checkpoint.restore("./ddpm_mnist.ckpt")
        pbar = tqdm(dataset)
        loss_ema = None
        for x in pbar:
            # pbar.set_description(f"Epoch {epoch}")
            # traning
            with tf.GradientTape() as tape:
                # get loss
                loss = ddpm(x)
                # loss function
                if loss_ema is None:
                    loss_ema = loss[0]
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss[0]
                # progress bar
                pbar.set_description(f"Epoch {epoch} | loss: {loss_ema:.4f}")
            # get gradient
            gradients = tape.gradient(loss_ema, ddpm.trainable_variables)
            # apply gradient
            optim.apply_gradients(zip(gradients, ddpm.trainable_variables))
        # sampling images
        if epoch % 100 == 0:
            print("start sampling")
            for i in range(10):
                xh = ddpm.sample(1, (64, 64, 3), device)
                xset = tf.concat([xh], axis=0)
                saveImage(xset, f"./train_cifar10TF_out/testOutput{epoch}/testOutput{epoch}/ddpm_sample_cifar{i}.png", nrow=1)
                print(f"save {i} image")
        # sampling images
        if epoch % 1000 == 0:
            xh = ddpm.sample(8, (64, 64, 3), device)
            xset = tf.concat([xh, x[:8]], axis=0)
            saveImage(xset, f"./train_cifar10TF_out/ddpm_sample_cifar{epoch+10}.png")


        if epoch in epochs:
            # save trained data from the model
            # checkpoint = tf.train.Checkpoint(optimizer=optim, model=ddpm)
            # checkpoint.save(file_prefix=f'train_cifar10weightTF/{epoch}_superminddpm_TF/{epoch}_train_cifar10.ckpt')


            if epoch in epochs:
                # # save trained data from the model
                # checkpoint = tf.train.Checkpoint(optimizer=optim, model=ddpm)
                # checkpoint.save(file_prefix=f'superweightTF/{epoch}_superminddpm_TF/{epoch}_superminddpm_TF.ckpt')

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

# runnning
if __name__ == "__main__":
    train_cifar10()
