# Object Detection with Zap

## Data setup

Zap expects to find your images and labels in specific folders. Whilst you can create this folder structure anywhere, the following guide will create it inside a `data` folder at the root of your project. With that said, this `data` folder **must** have an `images` folder, a `labels.json` file that is in the COCO format.

Place all of your images in the `images` folder. You do not need to split them into training, testing and validation sets; Zap will do that for you.

**Folder structure**

```
your_project/
  ├── data/
  │   ├── images/
  │   │   ├── image_01.jpg
  │   │   └── ...
  │   ├── labels.json
  │   └── predict/
  │       └── image_99.jpg
  ├── main.py
  └── config.yaml
```

`labels.json` needs to be in the COCO format.

## Configuration File

Almost everything to do with controlling Zap happens from the configuration file. Zap has already done lot of the base configuration for you. We just need to tell it 3 things:

- the **model** we want to use
- the **data** we want to load
- the **optimizer** we want to use

Here's an example configuration file that uses the state-of-the-art DETA architecture.

```yaml
model:
  class_path: zap.object_detection.models.Deta
  init_args:
    num_classes: 1
    lr: 1.0e-04
    lr_backbone: 1.0e-05
    weight_decay: 1.0e-04

data:
  class_path: zap.object_detection.data_modules.ObjectDetectionDataModule
  init_args:
    data_dir: data
    size:
      - 432
      - 768
    train_split: 0.7
    test_split: 0.2
    val_split: 0.1
    batch_size: 1
    num_workers: 0
    pin_memory: true
    shuffle: true
```

Currently we've hardcoded the optimizer for the DETA model due to some issues around running it from the config file.

Also note that transforms have not been tested with this model.

To learn more about this config file and how to use it, familiarise yourself with the Pytorch Lightning [docs](https://lightning.ai/docs/pytorch/stable/levels/advanced_level_15.html).
