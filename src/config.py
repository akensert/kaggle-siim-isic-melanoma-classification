from efficientnet.tfkeras import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
)

config = {
    'lr_max': 1e-4,
    'lr_min': 1e-6,
    'lr_decay_epochs': 16,
    'lr_warmup_epochs': 1,
    'lr_decay_power': 1,
    'n_epochs': 16,
    'label_smoothing': 0.05,
    'focal_loss': True,
    'tta': 1,
    'save_best': '../output/weights/',
    'pretrained_weights': 'noisy-student',
    'finetuned_weights': None,
}

fold_config = {
    0: {
        'engine': EfficientNetB0,
        'input_path': 'melanoma-512x512/',
        'input_shape': (512, 512, 3),
        'batch_size': 12,
    },
    1: {
        'engine': EfficientNetB1,
        'input_path': 'melanoma-512x512/',
        'input_shape': (512, 512, 3),
        'batch_size': 12,
    },
    2: {
        'engine': EfficientNetB2,
        'input_path': 'melanoma-512x512/',
        'input_shape': (512, 512, 3),
        'batch_size': 12,
    },
    3: {
        'engine': EfficientNetB3,
        'input_path': 'melanoma-384x384/',
        'input_shape': (384, 384, 3),
        'batch_size': 12,
    },
    4: {
        'engine': EfficientNetB4,
        'input_path': 'melanoma-384x384/',
        'input_shape': (384, 384, 3),
        'batch_size': 12,
    },
}
