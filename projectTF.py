import pathlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm
time_start = time.time()


# device = "/GPU:2" if tf.test.is_gpu_available else "/CPU:0"

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
    #print('sqrt_one_minus_alpha_cumprod_t', sqrt_one_minus_alpha_cumprod_t.shape, 'noise', noise.shape)
    noised_image = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    #print('noised_img', noised_img.shape)
    return noised_image,noise

def extract(a, t, x_shape): #可能问题很大
    batch_size = t.shape[0]
    out = tf.gather(a, t)
    #print(out)
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



def load_and_transform_dataset(train_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size))

    # Define a function to normalize images to [-1, 1]
    def normalize(input_image,x):
        input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
        return input_image

    train_dataset = train_dataset.map(normalize, tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(5000).prefetch(tf.data.AUTOTUNE)

    return train_dataset

dataset = load_and_transform_dataset(train_path)
dataset = dataset.apply(tf.data.experimental.unbatch())
dataset = dataset.batch(batch_size, drop_remainder = True)



def reverse_transform(image):
    #print('reverse_transform img ', image)
    # Reverse the normalization applied earlier
    image = (image + 1) /2
    image = image * 255
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    if len(image.shape) == 4:
        # Remove the batch dimension if it's there
        image = image[0]
    np = image.numpy()
    #print('img this is img after trans wow', np)
    image = Image.fromarray(np)
    plt.imshow(image)
    plt.axis('off')  # Hide the axis

import math

# tensorflow and pytorch have different tensor in graphic pytorch[N,C,H,W] tensorflow[N,H,W,C]
# N: batch size, H: height, W: width, C: channel
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
        #print(self.nam)
        #print('hhhhhhhhhhhhh',h.shape)
        # Down or Upsample
        return self.transform(h)


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
        # TODO: Double check the ordering here
        return embeddings

def relu(x):
    return tf.maximum(x, 0)

class RELU(tf.keras.layers.Layer):
    def __init__(self):
        super(RELU, self).__init__()

    def call(self, x, training=True):
        return relu(x)


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
            tf.keras.layers.Dense(units=time_emb_dim*2),
            RELU(),
            tf.keras.layers.Dense(units=time_emb_dim*2)]
        )

        # Initial projection
        self.conv0 = tf.keras.layers.Conv2D(filters = down_channels[0], kernel_size = 3, strides = (1, 1), padding = 'same', input_shape = (None, None, image_channels))#tensorflow set stride to 1 make sure same output size

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
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = tf.concat((x, residual_x), axis=-1) #dim = axis
            x = up(x, t)
        #print(x.shape)
        return self.output1(x)


model = SimpleUnet()
# next two line is for load the pretrained data into model
tqdm(model.load_weights('200_pixel_weights.index'))
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
    #print('reverse_transform img 11111', x)
    betas_t = extract(beta, t, x.shape)
    #print('reverse_transform img 222222', betas_t)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alpha_cumprod, t, x.shape
    )
    #print('reverse_transform img 11111', sqrt_one_minus_alphas_cumprod_t)
    sqrt_recip_alphas_t = extract(one_over_sqrt_alpha, t, x.shape)
    #print('reverse_transform img 22222', sqrt_recip_alphas_t)
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    #print('reverse_transform img 33333', model_mean)
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    #print('reverse_transform img 1111111', posterior_variance_t)
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        #print('reverse_transform img 33333  000000', model_mean)
        return model_mean
    else:
        noise = tf.random.normal(shape=tf.shape(x))
        a = tf.sqrt(posterior_variance_t)
        ret = model_mean + tf.sqrt(posterior_variance_t) * noise #can not use tf.math
        #print('reverse_transform img 33333', a)
        return ret


def sample_plot_image(milestone):
    # Sample noise
    img = tf.random.normal((1, img_size, img_size,3))
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 4
    stepsize = int(steps / num_images)
    for i in range(0, steps)[::-1]:
        t = tf.fill((1,), i)
        #print('reverse_transform img ', img)
        img = sample_timestep(img, t) # problem !!!!!!!!
        #print('reverse_transform img ', img)
        # Edit: This is to maintain the natural range of the distribution
        img = tf.clip_by_value(img, -1.0, 1.0)
        #print('reverse_transform img 11111', img)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            reverse_transform(img.numpy()) # 等同于 detach in pytorch 可以断开 和 tensor的连接
    plt.savefig(f'1000output/out-{milestone}.png', dpi=300)
    plt.close()



epochs = 500 # Try more!

def set_key(key):
    np.random.seed(key)


def generate_timestamp(key, num):
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=steps, dtype=tf.int32)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


for epoch in range(epochs):
    for step, batch in enumerate(dataset):
        # calculate gradient
        with tf.GradientTape() as tape:
            t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=steps, dtype=tf.int32)
            #print('this is batch', batch.shape)
            loss = get_loss(model, batch, t)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 1 == 0 and step == 0:
            print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.numpy()} ")# numpy is same as item it will break the link between tensors
            sample_plot_image(epoch)

# save trained data from the model
# model.save_weights('200_pixel_weights')

time_end = time.time()
time_sum = time_end - time_start
print(time_sum)
