import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import pathlib
import math
from tqdm import tqdm
import numpy as np
from scipy import linalg
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import img_to_array


# load a model from tensorflow
inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                weights="imagenet",
                                pooling='avg')


#reform the picture
img_size = 75
batch_size = 64
BUFFER_SIZE = 1000

# load and transform dataset
def load_and_transform_dataset(train_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        seed=123,
        image_size=(img_size, img_size))

    # Define a function to normalize images to [-1, 1]
    def normalize(input_image,x):
        input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
        return input_image

    # apply normalization
    train_dataset = train_dataset.map(normalize, tf.data.AUTOTUNE)
    # shuffle the data
    train_dataset = train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)

    return train_dataset

# FID calculation for gray images
# def compute_embeddings(dataloader, count):
#     image_embeddings = []
#
#     for _ in tqdm(range(count)):
#         images = next(iter(dataloader))
#         images = preprocessImage(images)
#         embeddings = inception_model.predict(images)
#
#         image_embeddings.extend(embeddings)
#
#     return np.array(image_embeddings)

# calculate embedding by using inception
def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        # images = preprocessImage(images)
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

def calculateFID(real, generated):
    # calculate mean and covariance statistics
    mean1, cov1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mean2, cov2 = generated.mean(axis=0), np.cov(generated, rowvar=False)
    # calculate sum squared difference between means
    sdifmean = np.sum((mean1 - mean2) ** 2.0)
    # calculate sqrt of product between cov
    newcov = linalg.sqrtm(cov1.dot(cov2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(newcov):
        newcov = newcov.real
    # calculate score
    fid = sdifmean + np.trace(cov1 + cov2 - 2.0 * newcov)
    return fid

def preprocessImage(img):
    img = tf.image.resize(img, [75, 75])
    x = img.numpy()
    x = tf.tile(x, [1, 1, 1, 3])  # copy channel
    x = x / 127.5 - 1.0  # change range to [-1, 1]
    return x
# list of epochs
epochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# list of FID
fidList = []
for i in epochs:
    # self define normalization
    def normalize(image,x):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 1.0
        return image

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

    # calculate 1/4 datas
    quarter_dataset_size = dataset_size // 60

    # get 1/4 data from datasets
    dataset = dataset.take(quarter_dataset_size)
    dataloader = dataset.cache()
    dataloader = dataloader.shuffle(buffer_size=dataset_size)
    dataloader = dataloader.map(normalize)
    dataloader = dataloader.batch(64, drop_remainder = True)

    train_path = pathlib.Path('./front')
    gen_path = pathlib.Path(f'train_cifar10PT_out/testOutput{i}')

    # train dataset
    train_dataset = load_and_transform_dataset(train_path)
    train_dataset = train_dataset.apply(tf.data.experimental.unbatch())
    train_dataset = train_dataset.batch(batch_size, drop_remainder = True)

    # generate dataset
    gen_dataset = load_and_transform_dataset(gen_path)
    gen_dataset = gen_dataset.apply(tf.data.experimental.unbatch())
    gen_dataset = gen_dataset.batch(1, drop_remainder = True)

    count = math.ceil(10000/batch_size)
    count2 = math.ceil(10000 / 1000)
    # compute embeddings for real images
    realembeddings = compute_embeddings(train_dataset, count)

    # compute embeddings for generated images
    genembeddings = compute_embeddings(gen_dataset, count2)

    # get fid score
    fid = calculateFID(realembeddings, genembeddings)

    fidList.append(fid)

print(fidList)