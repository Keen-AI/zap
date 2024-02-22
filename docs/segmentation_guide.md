# Segmentation with Zap

We currently only support **binary** segmentation, but will soon be adding multi-class support for these models.

As such, the masks you use for training here should be "black and white" PNGs (black for background, white for class label)

## Data setup

Zap expects to find your images and labels in specific folders. Whilst you can create this folder structure anywhere, the following guide will create it inside a `data` folder at the root of your project. With that said, this `data` folder **must** have an `images` folder and a `masks` folder.

Place all of your images in the `images` folder and all of the accompanying masks in the `masks` folder. You do not need to split them into training, testing and validation sets; Zap will do that for you.

**Folder structure**

```
your_project/
├── data/
│   ├── images/
│   │   ├── image_01.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── image_01.png
│   │   └── ...
│   └── predict/
│       └── image_99.jpg
├── main.py
└── config.yaml
```

## Configuration File

Almost everything to do with controlling Zap happens from the configuration file. Zap has already done lot of the base configuration for you. We just need to tell it 3 things:

- the **model** we want to use
- the **data** we want to load
- the **optimizer** we want to use

Here's an example configuration file that uses ResNet34 with some basic augmentations/transforms.

```yaml
model:
  class_path: zap.segmentation.models.UNet
  init_args:
    num_classes: 1
    encoder_name: efficientnet-b1
    encoder_weights: imagenet
    encoder_depth: 5
    activation: sigmoid

data:
  class_path: zap.segmentation.data_modules.SegmentationDataModule
  init_args:
    data_dir: data
    transforms:
      - class_path: torchvision.transforms.ToTensor
      - class_path: torchvision.transforms.Resize
        init_args:
          size:
            - 256
            - 256
          antialias: true
    train_split: 0.7
    test_split: 0.2
    val_split: 0.1
    batch_size: 2
    num_workers: 0
    pin_memory: true
    shuffle: true

optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001234
    betas:
      - 0.9
      - 0.999
    eps: 1.0e-08
    weight_decay: 0.0
    amsgrad: false
    foreach: null
    maximize: false
    capturable: false
    differentiable: false
    fused: null
```

In this example we're using the `Adam` optimizer with a learning rate of 0.001. For more optimizers you can explore the `torch.optim` module (or Pytorch docs), **but keep in mind that the Stochastic Weight Averaging technique does not support all optimizers.**

To learn more about this config file and how to use it, familiarise yourself with the Pytorch Lightning [docs](https://lightning.ai/docs/pytorch/stable/levels/advanced_level_15.html).
