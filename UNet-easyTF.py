
from typing import Dict, Tuple
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
import psutil
import os
import subprocess

# ddpm_schedules get all the parameters for future calculation
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, tf.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * tf.range(0, T + 1, dtype=tf.float32) / T + beta1
    sqrt_beta_t = tf.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = tf.math.log(alpha_t)
    alphabar_t = tf.math.exp(tf.cumsum(log_alpha_t, axis=0))

    sqrtab = tf.sqrt(alphabar_t)
    oneover_sqrta = 1 / tf.sqrt(alpha_t)

    sqrtmab = tf.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

# block small UNet
blk = lambda ic, oc: tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = oc, kernel_size = 7, padding='same'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.LeakyReLU(alpha=0.01),
])


class DummyEpsModel(tf.keras.Model):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = tf.keras.Sequential([  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            tf.keras.layers.Conv2D(filters=n_channel, kernel_size=3, padding='same'),
        ])

    def call(self, x, t) -> tf.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)

class DDPM(tf.keras.Model):
    def __init__(
        self,
        eps_model: tf.keras.Model,
        betas: Tuple[float, float],
        n_T: int,
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            setattr(self, k, tf.Variable(v, trainable=False),)

        self.n_T = n_T
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = tf.random.uniform(shape=[tf.shape(x)[0]], minval=1, maxval=self.n_T, dtype=tf.int32)  # t ~ Uniform(0, n_T)
        eps = tf.random.normal(shape=tf.shape(x))  # eps ~ N(0, 1)
        gathered_sqrtab = tf.gather(self.sqrtab, _ts, axis=0)
        gathered_sqrtmab = tf.gather(self.sqrtmab, _ts, axis=0)
        x_t = (
            gathered_sqrtab[:, None, None, None] * x
            + gathered_sqrtmab[:, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        return self.mse(eps, self.eps_model(x_t, _ts / self.n_T)),

    def sample(self, n_sample: int, size, device) -> tf.Tensor:

        x_i = tf.random.normal([n_sample] + list(size))  # x_T ~ N(0, 1)

        # sampling process
        for i in range(self.n_T, 0, -1):
            z = tf.random.normal([n_sample] + list(size)) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

def saveImage(images, filename, nrow=4):
    # image 0 - 1
    images = images + 0.5
    # change to numpy can use by plt
    images = images.numpy()
    # number of picture in a row
    fig, axes = plt.subplots(nrows=nrow, ncols=nrow, figsize=(nrow, nrow), dpi=28)
    # make it can use for loop
    if nrow == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # go trough all the pic
    for img, ax in zip(images, axes):
        # delete additional 1 in tensor and the color is in gray
        ax.imshow(img.squeeze(),cmap='gray')
        ax.axis('off')
    # remove all the space between pictures
    plt.tight_layout(pad=0)
    plt.savefig(filename)
    plt.close()



def train_mnist(n_epoch: int = 201, device="cuda:0") -> None:
    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    # save model by check point
    # checkpoint = tf.train.Checkpoint(model=ddpm)

    def normalize(image,x):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 1.0
        return image,x

    dataset, dataset_info = tfds.load(
        'mnist',
        split='train',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir="./data"
    )
    # get size of dataset
    dataset_size = dataset_info.splits['train'].num_examples

    # get quarter size of dataset
    smallerdataset_size = dataset_size // 4

    # apply all the changes
    dataset = dataset.take(smallerdataset_size)
    dataloader = dataset.cache()
    dataloader = dataloader.shuffle(buffer_size=smallerdataset_size)
    dataloader = dataloader.map(normalize)
    dataloader = dataloader.batch(128, drop_remainder = True)
    optim = tf.keras.optimizers.Adam(learning_rate=2e-4)

    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Try more!

    timelsit = []
    gpulist = []
    # for i in epochs:
    time_start = time.time()
    # load data
    # checkpoint = tf.train.Checkpoint(optimizer=optim, model=ddpm)
    for i in epochs:
        for epoch in range(1, n_epoch):
            # load data
            # restore = checkpoint.restore(f'superweightTF/{i}_superminddpm_TF/{i}_superminddpm_TF.ckpt-1')

            #training
            pbar = tqdm(dataloader)
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

            # get sample images from ddpm
            xh = ddpm.sample(1, (28, 28, 1),device)
            saveImage(xh, f"./superminddpmTF/testOutput{i}/testOutput{i}/ddpm_sample_{epoch}.png", nrow=1)
            print(f"File {i} | Epoch {epoch}")

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
    train_mnist()
