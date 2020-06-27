augmentor = (
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
