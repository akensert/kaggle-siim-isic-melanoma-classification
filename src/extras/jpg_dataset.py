import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from albumentations import *


augmentor = (
    Compose([
        OneOf([
            ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=90,
                p=0.5),
            NoOp()
        ]),
        OneOf([
            RandomSizedCrop(
                min_max_height=(256, 384),
                height=384,
                width=384,
                w2h_ratio=0.85,
                p=0.5),
            Downscale(
                scale_min=0.25,
                scale_max=0.25,
                p=0.5),
            NoOp()
        ]),
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5),
            RandomGamma(
                gamma_limit=(80, 120),
                p=0.5),
            NoOp()
        ]),
        RandomRotate90(
            p=0.5),
        Flip(
            p=0.5),
        Transpose(
            p=0.5),
    ])
)

def tf_map_decorator(func):
    '''Decorate python functions with this
    decorator: Enables arbitrary Python
    code to work with/in the TF graph
    '''
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[a.dtype for a in args]
        )
    return wrapper

def _read_input(path, target):
    '''Reads and decodes the image from input 'path'.
    Input 'target' is also passed to the function for
    convenience (to work well with dataset.map(..))
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, target

def _resize_input(x, y, target_size, resize_pad):
    '''Resizes input image 'x' to target_size. Also takes an extra
    argument 'pad' which keeps the aspect ratio on resize by
    padding the image.
    '''
    if tf.reduce_any(tf.shape(x) != target_size):
        if resize_pad:
            x = tf.image.resize_with_pad(x, *target_size[:-1], method='bilinear')
        else:
            x = tf.image.resize(x, target_size[:-1], method='bilinear')
        x = tf.cast(x, dtype=tf.uint8)
    return x, y

@tf_map_decorator
def _augment_input(image, target):
    '''Using albumentations augmentations wrapped in
    tf.py_functions is very convenient. However, for
    better performance, consider using TF operations
    instead to augment the image data
    '''
    image = augmentor(image=image.numpy())['image']
    return image, target

def _preprocess_input(image, target):
    '''Simple preprocessing of the input,
     before being fed to the neural network
    '''
    image = tf.cast(image, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)
    target = tf.expand_dims(target, -1)
    image /= 255.
    return image, target


class InitializeJPGDataset:

    def __init__(self,
                 path,
                 keys=['image', 'target'],
                 input_shape=(384, 384, 3),
                 batch_size=16,
                 resize_pad=False,
                 augment=None,
                 shuffle=None,
                 cache=None):

        self.path = path
        self.keys = keys
        self.has_target = True if 'train' in self.path.split('/')[-1] else False
        self.image_paths = glob.glob(self.path)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.resize_pad = resize_pad
        self.augment = augment
        self.shuffle = shuffle
        self.cache = cache

    def get(self, dataframe):

        if self.cache:
            if not(os.path.isdir('tmp/')):
                os.mkdir('tmp/')
            else:
                files = glob.glob('tmp/*')
                for file in files:
                    os.remove(file)

            if isinstance(self.cache, str):
                cache_path = 'tmp/' + cache
            else:
                cache_path = ''

        image_paths = path.rstrip('/') + '/' + dataframe.image_name + '.jpg'
        if self.has_target == False:
            targets = np.zeros(len(image_paths), dtype=np.int64)
        else:
            targets = dataframe.target
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, targets))

        dataset = dataset.map(
            _read_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            lambda x, y: _resize_input(
                x, y, target_size=self.input_shape, resize_pad=self.resize_pad),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.cache:
            dataset = dataset.cache(cache_path)

        if self.shuffle:
            dataset = dataset.shuffle(1024)

        if self.augment:
            dataset = dataset.map(_augment_input, tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(
            _preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
