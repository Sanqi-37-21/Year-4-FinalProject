import pathlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Model
from PIL import Image
from tqdm import tqdm
import os
import subprocess


#beta increase from begin to end
def beta_increase(steps):
    beta_start = 0.0001
    beta_end = 0.02
    return tf.linspace(beta_start, beta_end, steps)

#forward diffusion
def forward_step(x0,t):
    noise = np.random.normal(size=x0.shape)
    sqrt_alpha_cumprod_t = extract(sqrt_alpha_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(
        sqrt_one_minus_alpha_cumprod, t, x0.shape
    )
    noised_image = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return noised_image,noise

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = tf.gather(a, t)
    out_reshaped = tf.reshape(out, [batch_size] + [1] * (len(x_shape) - 1))
    return out_reshaped


#number of steps
steps = 100

#initial beta
beta = beta_increase(steps)
#define alpha
alpha = 1-beta
#cumprod of alpha
alpha_cumprod = tf.math.cumprod(alpha, axis = 0)
#sqrt(bar(a))
sqrt_alpha_cumprod = tf.math.sqrt(alpha_cumprod)
alpha_cumprod_prev = tf.pad(alpha_cumprod[:-1], paddings=[[1, 0]])
#sqrt(1-bar(a))
sqrt_one_minus_alpha_cumprod = tf.math.sqrt(1-alpha_cumprod)
#1/sqrt(a)
one_over_sqrt_alpha = tf.math.sqrt(1.0 / alpha)
#b/sqrt(1-bar(a))
beta_over_sqrt_one_minus_alpha_cumprod= beta/sqrt_one_minus_alpha_cumprod
posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

train_path = pathlib.Path('./front')
#reform the picture
img_size = 64
batch_size = 64
BUFFER_SIZE = 1000


# load_trans_dataset in PyTorch
def load_and_transform_dataset(train_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        validation_split=0.01,
        subset="training",
        seed=123,
        image_size=(img_size, img_size))

    # Define a function to normalize images to [-1, 1]
    def normalize(image,x):
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image
    # apply the normalization
    train_dataset = train_dataset.map(normalize, tf.data.AUTOTUNE)
    # shuffle the data
    train_dataset = train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)

    return train_dataset

# laod and batch data
dataset = load_and_transform_dataset(train_path)
dataset = dataset.apply(tf.data.experimental.unbatch())
dataset = dataset.batch(batch_size, drop_remainder = True)


# reverse_trans in PyTorch
def reverse_transform(image):
    # Reverse the normalization applied earlier
    image = (image + 1) /2
    image = image * 255
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    if len(image.shape) == 4:
        # Remove the batch dimension if it's there
        image = image[0]
    np = image.numpy()
    image = Image.fromarray(np)
    plt.imshow(image)
    plt.axis('off')  # Hide the axis

import math

# tensorflow and pytorch have different tensor in graphic pytorch[N,C,H,W] tensorflow[N,H,W,C]
# N: batch size, H: height, W: width, C: channel
# Block class in PyTorch
class Block(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = tf.keras.layers.Dense(units=out_ch, input_shape=(time_emb_dim,)) # dense is linear in pytorch
        if up:
            #self.conv1 = tf.keras.layers.Conv2D(2 * in_ch, out_ch, 1, padding='same')
            self.conv1 = tf.keras.layers.Conv2D(filters= out_ch,kernel_size=3 , strides=1, padding='same', input_shape=(None, None, 2*in_ch) )
            self.transform = tf.keras.layers.Conv2DTranspose(filters = out_ch, kernel_size = 4, strides = 2, padding = 'same')
            self.nam = 'up'
        else:
            #self.conv1 = tf.keras.layers.Conv2D(in_ch, out_ch, 1, padding='same')
            self.conv1 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, strides = 1, padding = 'same', input_shape = (None, None, in_ch))
            self.transform = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 4, strides = 2, padding = 'same', input_shape = (None, None, out_ch))
            self.nam = 'down'
        #self.conv2 = tf.keras.layers.Conv2D(out_ch, out_ch, 1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, strides = (1, 1), padding = 'same', input_shape = (None, None, out_ch))
        self.bnorm1 = tf.keras.layers.BatchNormalization(axis=-1) #tensorflow didn't define axis as default
        self.bnorm2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, t, training=True):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        #time_emb = time_emb[(...,) + (None,) * 2]
        time_emb = tf.expand_dims(tf.expand_dims(time_emb, 1), 2)
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

# SinusoidalPositionEmbeddings class in PyTorch
class SinusoidalPositionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, time, training=True):
        time = tf.cast(time, tf.float32)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = tf.exp(tf.range(half_dim, dtype=tf.float32) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        return embeddings

# nn.ReLU()
# if x > 0, x=x, else x=0
def relu(x):
    return tf.maximum(x, 0)

class RELU(tf.keras.layers.Layer):
    def __init__(self):
        super(RELU, self).__init__()

    def call(self, x, training=True):
        return relu(x)

# SimpleUnet class in PyTorch
class SimpleUnet(tf.keras.Model):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = tf.keras.Sequential([
            SinusoidalPositionEmbeddings(time_emb_dim),
            # nn.Linear
            tf.keras.layers.Dense(units=time_emb_dim * 2),
            RELU(),]
        )

        # Initial projection PyTorch API Conv2d
        self.conv0 = tf.keras.layers.Conv2D(filters = down_channels[0], kernel_size = 3, strides = (1, 1), padding = 'same', input_shape = (None, None, image_channels))#tensorflow set stride to 1 make sure same output size

        # Downsample
        self.downs = []
        for i in range(len(down_channels) - 1):
            self.downs.append(Block(down_channels[i], down_channels[i + 1], time_emb_dim))

        # Upsample
        self.ups = []
        for i in range(len(up_channels) - 1):
            self.ups.append(Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True))

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output1 = tf.keras.layers.Conv2D(filters = out_dim, kernel_size = 1, strides = (1, 1), input_shape = (None, None, 64))

    def call(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        # down sample
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        # up sample
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = tf.concat((x, residual_x), axis=-1) #dim = axis
            x = up(x, t)
        # return output
        return self.output1(x)


model = SimpleUnet()
# next two line is for load the pretrained data into model
# tqdm(model.load_weights('200_pixel_weights'))
#print("Num params: ", sum(p.numel() for p in model.parameters()))

def get_loss(model, x_0, t):
    x_noisy, noise = forward_step(x_0, t)
    noise_pred = model(x_noisy, t)
    # rewrite F.l1_loss
    absolute_difference = tf.abs(noise - noise_pred)
    loss = tf.reduce_mean(absolute_difference)
    return loss



def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = extract(beta, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alpha_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(one_over_sqrt_alpha, t, x.shape)
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = tf.random.normal(shape=tf.shape(x))
        a = tf.sqrt(posterior_variance_t)
        ret = model_mean + tf.sqrt(posterior_variance_t) * noise #can not use tf.math
        return ret

# use for sampling
def sample_plot_image(milestone,filenumber):
# use for training
# def sample_plot_image(milestone):
    # Sample noise
    img = tf.random.normal((1, img_size, img_size,3))
    plt.figure(figsize=(1, 1))
    plt.axis('off')

    # number of images
    num_images = 1
    # stepsize of images
    stepsize = int(steps / num_images)
    for i in range(0, steps)[::-1]:
        t = tf.fill((1,), i)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = tf.clip_by_value(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            # PyTorch detach API  which can disconnect the link with tensor
            reverse_transform(img.numpy())
    # {filenumber}
    # since bbox_inches='tight' will smaller the size so make dpi=83 to make sure output image have size 64*64
    # for sampling
    plt.savefig(f'TFNewStep100Image/testOutput{filenumber}/testOutput{filenumber}/out-{milestone}.png', dpi=84,bbox_inches='tight', pad_inches=0)
    # for training
    # plt.savefig(f'TensorFlowOutput/testOutput/out-{milestone}.png', dpi=84,bbox_inches='tight', pad_inches=0)
    plt.close()



epochs = [400,500] # Try more!

# time list
timelist = []
gpulist = []
for i in epochs:
    # record start time
    time_start = time.time()
    model = SimpleUnet()
    # initialize the model
    start_input = tf.random.normal([1, 64, 64, 3])
    _ = model(start_input,tf.constant([0], dtype=tf.int32))
    tqdm(model.load_weights(f'{i}_pixel_weights_step100.h5'))
    # optimizer in tensorflow
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    # range(500)
    # for epoch in range(500):
    for epoch in range(1,201):
        for step, batch in enumerate(dataset):
            # using for training
            # # calculate gradient
            # with tf.GradientTape() as tape:
            #     t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=steps, dtype=tf.int32)
            #     #print('this is batch', batch.shape)
            #     loss = get_loss(model, batch, t)
            # # gradient update in tensorflow
            # gradients = tape.gradient(loss, model.trainable_variables)
            # optim.apply_gradients(zip(gradients, model.trainable_variables))

            # using for test FID
            t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=steps, dtype=tf.int32)
            loss = get_loss(model, batch, t)
            if epoch % 1 == 0 and step == 0:
                print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.numpy()} ")# numpy is same as item it will break the link between tensors
                # using for FID
                sample_plot_image(epoch,i)
                # using for traning
                # sample_plot_image(epoch)
    # using for training
    if epoch in epochs:
        # save trained data from the model
        model.save_weights(f'{epoch}_pixel_weights_step100.h5')

        time_end = time.time()
        time_sum = time_end - time_start
        print(time_sum)
        timelist.append(time_sum)

        # using nvidia-smi command get GPU memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)
        # decode the output
        output = result.stdout.decode('utf-8')

        # get the memory used
        for line in output.strip().split('\n'):
            total, used = line.split(', ')
            gpulist.append({"total_memory_MB": int(total), "used_memory_MB": int(used)})

print(timelist)
print(gpulist)
