import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
import glob

from model import DistributedModel
from generator import get_dataset
from optimizer import get_optimizer
from config import config, fold_config

mixed_precision = False
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    mixed_precision = True
    # turn on mixed precision
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")


path = '../input/siim-isic-melanoma-classification/'

input_path = '../input/'
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')

submission_data = pd.read_csv(path + 'sample_submission.csv')
test_data['target'] = 0
print("test shape =", test_data.shape)
print(test_data.head(3))
print("\ntrain shape =", train_data.shape)
print(train_data.head(3))



splits = model_selection.KFold(
    len(fold_config), True, 42).split(X=range(15))

with strategy.scope():

    valid_preds_accum, test_preds_accum, test_names_accum = list(), list(), list()

    for fold, (train_idx, valid_idx) in enumerate(splits):

        optimizer = get_optimizer(
            steps_per_epoch=33126//fold_config[fold]['batch_size'], # rough estimation
            lr_max=config['lr_max'],
            lr_min=config['lr_min'],
            decay_epochs=config['lr_decay_epochs'],
            warmup_epochs=config['lr_warmup_epochs'],
            power=config['lr_decay_power']
        )

        dist_model = DistributedModel(
            engine=fold_config[fold]['engine'],
            input_shape=fold_config[fold]['input_shape'],
            pretrained_weights=config['pretrained_weights'],
            finetuned_weights=config['finetuned_weights'],
            batch_size=fold_config[fold]['batch_size'],
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=mixed_precision,
            label_smoothing=config['label_smoothing'],
            tta=config['tta'],
            focal_loss=config['focal_loss'],
            save_best=config['save_best'])


        tfrec_paths = np.asarray(
            glob.glob(input_path+fold_config[fold]['input_path']+'train*'))
        test_paths = glob.glob(
            input_path+fold_config[fold]['input_path']+'test*')
        train_paths = tfrec_paths[train_idx]
        valid_paths = tfrec_paths[valid_idx]

        train_ds = get_dataset(
            train_paths, fold_config[fold]['batch_size'], augment='heavy', shuffle=True)
        valid_ds = get_dataset(
            valid_paths, fold_config[fold]['batch_size'], augment=False)
        test_ds = get_dataset(
            test_paths, fold_config[fold]['batch_size'], augment=False)

        valid_preds, _, test_preds, test_names = dist_model.fit_and_predict(
            fold=fold,
            epochs=config['n_epochs'],
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
        )

        valid_preds_accum.append(valid_preds)
        test_preds_accum.append(test_preds)
        test_names_accum.append(test_names)

        dist_model.reset_weights()

final_preds = np.average(test_preds_accum, axis=0) # weights=[..] for weighted avg
final_preds_map = dict(zip(test_names_accum[0].astype('U13'), final_preds))
submission_data['target'] = submission_data.image_name.map(final_preds_map)
submission_data.to_csv('submission.csv', index=False)
