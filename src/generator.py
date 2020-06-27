import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import os
from albumentations import *
from PIL import Image


augmentor_heavy = (
    Compose([
        OneOf([
            ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=90,
                p=0.5),
            ElasticTransform(
                alpha=601,
                sigma=20,
                alpha_affine=10,
                p=0.3),
            GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3),
            RandomGridShuffle(
                grid=(3, 3),
                p=0.3),
            OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.2,
                p=0.3),
            NoOp()
        ]),
        OneOf([
            CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.5),
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
            MedianBlur(
                blur_limit=7,
                p=0.5),
            GaussianBlur(
                blur_limit=7,
                p=0.5),
            Blur(
                blur_limit=7,
                p=0.5),
            GlassBlur(
                sigma=0.7,
                max_delta=4,
                iterations=2,
                p=0.5),
            RandomFog(
                p=0.5),
            Posterize(
                num_bits=4,
                p=0.5),
            NoOp()
        ]),
        OneOf([
            GaussNoise(
                var_limit=(10.0, 100.0),
                p=0.3),
            ISONoise(
                color_shift=(0.05, 0.1),
                intensity=(0.1, 0.5),
                p=0.3),
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
        OneOf([
            RGBShift(
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.5),
            HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.5),
            FancyPCA(
                alpha=0.5,
                p=0.2),
            ChannelDropout(
                channel_drop_range=(1, 1),
                p=0.2),
            ToGray(
                p=0.2),
            ToSepia(
                p=0.2),
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

augmentor_light = (
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
    """Decorate python functions with this
    decorator: Enables arbitrary Python
    code to work with/in the TF graph
    """
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[a.dtype for a in args]
        )
    return wrapper

@tf_map_decorator
def _apply_light_augmentation(image, target):
    """Using albumentations augmentations wrapped in
    tf.py_functions is very convenient. However, for
    better performance, consider using TF operations
    instead to augment the image data
    """
    image = augmentor_light(image=image.numpy())['image']
    return image, target

@tf_map_decorator
def _apply_heavy_augmentation(image, target):
    """Using albumentations augmentations wrapped in
    tf.py_functions is very convenient. However, for
    better performance, consider using TF operations
    instead to augment the image data
    """
    image = augmentor_heavy(image=image.numpy())['image']
    return image, target

def _read_input(path, target, target_size, pad=False):
    """Reads, decodes and resizes the image.
    'target' is also passed to the function for
    convenience (to work well with dataset.map(..))
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    if pad:
        image = tf.image.resize_with_pad(image, *target_size[:-1], method='bilinear')
    else:
        image = tf.image.resize(image, target_size[:-1], method='bilinear')
    image = tf.cast(image, dtype=tf.uint8)
    return image, target

def _preprocess_input(image, target):
    """Simple preprocessing of the input,
    before being fed to the neural network
    """
    image = tf.cast(image, tf.float32)
    target = tf.cast(target, tf.float32)
    image /= 255.
    return image, target

def create_dataset(path,
                   dataframe,
                   input_shape,
                   batch_size,
                   augment=None,
                   shuffle=None,
                   cache=None):

    if cache:
        if not(os.path.isdir('tmp/')):
            os.mkdir('tmp/')
        else:
            files = glob.glob('tmp/*')
            for file in files:
                os.remove(file)

        if isinstance(cache, str):
            cache_path = 'tmp/' + cache
        else:
            cache_path = ''

    if shuffle:
        dataframe = dataframe.sample(frac=1.0, replace=False)

    image_paths = path + dataframe.image_name + '.jpg'
    targets = np.expand_dims(dataframe.target, -1)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, targets))

    dataset = dataset.map(
        lambda x, y: (
            _read_input(x, y, target_size=input_shape)
        ), tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache(cache_path)

    if augment == 'heavy':
        dataset = dataset.map(_apply_heavy_augmentation, tf.data.experimental.AUTOTUNE)
    elif augment == 'light':
        dataset = dataset.map(_apply_light_augmentation, tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_preprocess_input, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
