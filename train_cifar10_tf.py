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
import tensorflow_datasets as tfds
from PIL import Image

from mindiffusion.unettf import NaiveUnet
from mindiffusion.ddpmtf import DDPM

def saveImage(images, filename, nrow=4):
    # image 0 - 1
    # images = ((images + 1.0) / 2.0) * 255
    # images = tf.clip_by_value(images, 0, 255)
    # images = tf.cast(images, tf.uint8)
    # if len(images.shape) == 4:
    #     # Remove the batch dimension if it's there
    #     images = images[0]
    # # change to numpy can use by plt
    # images = images.numpy()
    # # 4*4
    # fig, axes = plt.subplots(nrows=nrow, ncols=nrow, figsize=(nrow, nrow))
    # # make it can use for loop
    # axes = axes.flatten()
    images = images * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    # change to numpy can use by plt
    images = images.numpy()
    # 4*4
    fig, axes = plt.subplots(nrows=nrow, ncols=nrow, figsize=(nrow, nrow))
    # make it can use for loop
    axes = axes.flatten()
    # go trough all the pic
    for img, ax in zip(images, axes):
        # delete additional 1 in tensor and the color is in gray
        ax.imshow(img)
        ax.axis('off')
    # remove all the space between pictures
    plt.tight_layout(pad=0)
    plt.savefig(filename)
    plt.close()

def train_cifar10(
    n_epoch: int = 100, device: str = "cuda:1", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    def normalize(image, x):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
        return image, x

    dataset, dataset_info = tfds.load(
        'cifar10',
        split='train',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir="./data"
    )
    dataset = dataset.map(normalize)
    # change to smaller batch 好像tensorflow对于gpu的利用效率要低
    dataset = dataset.batch(64, drop_remainder=True)
    optim = tf.keras.optimizers.Adam(learning_rate=1e-5)

    for i in range(n_epoch):
        # if Path("./ddpm_mnist.ckpt").exists():
        #     checkpoint.restore("./ddpm_mnist.ckpt")
        pbar = tqdm(dataset)
        loss_ema = None
        for x, _ in pbar:
            with tf.GradientTape() as tape:
                loss = ddpm(x)
                if loss_ema is None:
                    loss_ema = loss[0]
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss[0]
                pbar.set_description(f"loss: {loss_ema:.4f}")
            gradients = tape.gradient(loss_ema, ddpm.trainable_variables)
            optim.apply_gradients(zip(gradients, ddpm.trainable_variables))

        xh = ddpm.sample(8, (32, 32, 3), device)
        xset = tf.concat([xh, x[:8]], axis=0)
        saveImage(xset, f"./contents/ddpm_sample_cifar{i}.png")

        # save model
        # checkpoint.save(f"./ddpm_mnist.ckpt")


if __name__ == "__main__":
    train_cifar10()
