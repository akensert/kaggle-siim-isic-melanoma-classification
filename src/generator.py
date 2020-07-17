import tensorflow as tf
import math

def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')

    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    rot_mat = get_3x3_mat([c1,    s1,   zero ,
                           -s1,   c1,   zero ,
                           zero,  zero, one ])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_mat = get_3x3_mat([one,  s2,   zero ,
                             zero, c2,   zero ,
                             zero, zero, one ])

    zoom_mat = get_3x3_mat([one/hzoom, zero,      zero,
                            zero,      one/wzoom, zero,
                            zero,      zero,      one])

    shift_mat = get_3x3_mat([one,  zero, hshift,
                             zero, one,  wshift,
                             zero, zero, one   ])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )

def _spatial_transform(image,
                       rotation=180.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):

    dim = tf.gather(tf.shape(image), 0)

    xdim = dim % 2 # fix for size 331

    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # list destination pixel indices
    x = tf.repeat(tf.range(dim//2, -dim//2,-1), dim)
    y = tf.tile(tf.range(-dim//2, dim//2), [dim])
    z = tf.ones([dim*dim], dtype='int32')
    idx = tf.stack([x,y,z])

    # rotate destination pixels onto origin pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    idx2 = tf.clip_by_value(idx2, -dim//2+xdim+1, dim//2)

    # find origin pixel values
    idx3 = tf.stack([dim//2-idx2[0,], dim//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [dim, dim, 3])

def _pixel_transform(image,
                     saturation_delta=0.3,
                     contrast_delta=0.2,
                     brightness_delta=0.1):
    image = tf.image.random_saturation(
        image, 1-saturation_delta, 1+saturation_delta)
    image = tf.image.random_contrast(
        image, 1-contrast_delta, 1+contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image

def _flip_transpose(image):
    image = tf.image.random_flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.transpose(image, (1, 0, 2))
    return image

def _augment(features):
    features['image'] = _spatial_transform(features['image'])
    features['image'] = _flip_transpose(features['image'])
    features['image'] = _pixel_transform(features['image'])
    return features

def _parse_record(serialized, features_to_parse):

    features = {}
    for key in features_to_parse:
        if key == 'image_name' or key == 'image':
            features[key] = tf.io.FixedLenFeature(
                [], tf.string, default_value='')
        else:
            features[key] = tf.io.FixedLenFeature(
                [], tf.int64, default_value=0)
    example = tf.io.parse_single_example(
        serialized=serialized, features=features)

    extracted = {}
    for key in features_to_parse:
        if key == 'image':
            extracted[key] = tf.io.decode_jpeg(example[key], channels=3)
        else:
            extracted[key] = example[key]
    return extracted

def _preprocess_features(features):
    for key in features.keys():
        if key == 'image':
            features[key] = tf.cast(features[key], dtype=tf.float32) / 255.
        elif key == 'anatom_site_general_challenge':
            features[key] = tf.cast(tf.one_hot(features[key], 7), tf.float32)
        elif key == 'diagnosis':
            features[key] = tf.cast(tf.one_hot(features[key], 10), tf.float32)
        elif key == 'image_name':
            features[key] = tf.expand_dims(features[key], -1)
        else:
            features[key] = tf.expand_dims(
                tf.cast(features[key], dtype=tf.float32), -1)
    return features

def get_dataset(tfrec_paths,
                batch_size=16,
                augment=False,
                shuffle=False,
                cache=False):

    FEATURES_TO_PARSE = [
        'image', 'image_name', 'patient_id',
        'target', 'anatom_site_general_challenge',
        'sex', 'age_approx', 'diagnosis'
    ]

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

    dataset = tf.data.TFRecordDataset(
        filenames=tfrec_paths,
        num_parallel_reads=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda x: _parse_record(
            x, features_to_parse=FEATURES_TO_PARSE),
        tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache(cache_path)

    if shuffle:
        dataset = dataset.shuffle(1024)

    if augment:
        dataset = dataset.map(
            lambda features: _augment(features),
            tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(_preprocess_features, tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
