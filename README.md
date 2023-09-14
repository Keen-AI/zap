<div align="center">

<img alt="Lightning" src="logo.png" style="max-width: 100%;">
<br/>

**Zen AI Pipeline**

<small>🚧 IN DEVELOPMENT 🚧</small>

<small>Zap is a lightweight wrapper around Pytorch Lightning and MLFlow. It allows you to quickly train and test models for **classification**, **object detection** and **segmentation** without writing tonnes of boilerplate code.</small>

<hr>
</div>

### Install Zap

```
pip install -e TODO
```

### Example Usage

Using Zap is super simple. It's designed to be **config-driven**. That's why your `main.py` might look as simple as this:

```python
from zap import Zap

z = Zap()

if __name__ == '__main__':
    z.fit()
    z.test()
    preds = z.predict()
```

We can run this from the command line:

```
python main.py -c config.yaml
```

All of the parameters are defined in our `config.yaml` file (see Example Config section). In this file you define 3 key things:

- model
- optimizer
- data setup

But you can control a whole heap of settings that the Lightning API allows you to configure.

### Supported Models

Using different Zap models is easy.

```python
from zap.classification.models import ResNet34
from zap.object_detection.models import ResNet50_FasterRCNN
from zap.segmentation.models import UNet
```

Again, these can be plugged right into your `config.yaml`, not your `main.py`

#### Classification Models

- ResNet34

#### Object Detection Models

- TODO

#### Segmentation Models

- UNet

### Example Configuration Files

<details>
<summary>Classification</summary>

```yaml
# lightning.pytorch==2.0.6
seed_everything: true
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 32-true
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: demo
      log_model: true
      tags:
        example_tag_1: 123
        example_tag_2: 789
  callbacks:
    class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val_loss
      save_top_k: 1
      mode: min
  max_epochs: 10
  max_time: null
model:
  class_path: zap.classification.models.ResNet34
  init_args:
    num_classes: 12
    bias: true
data:
  class_path: zap.classification.data_modules.ClassificationDataModule
  init_args:
    data_dir: data/classification
    transforms:
      - class_path: torchvision.transforms.ToTensor
      - class_path: torchvision.transforms.Resize
        init_args:
          size:
            - 32
            - 32
          antialias: true
    train_split: 0.7
    test_split: 0.2
    val_split: 0.1
    batch_size: 64
    num_workers: 0
    pin_memory: true
    shuffle: true
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
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

</details>

<br>

<details>
<summary>Segmentation</summary>

```yaml
# lightning.pytorch==2.0.6
seed_everything: true
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 32-true
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: demo
      log_model: true
      tags:
        example_tag_1: 123
        example_tag_2: 789
  callbacks:
    class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val_loss
      save_top_k: 1
      mode: min
  max_epochs: 20
  max_time: null
  log_every_n_steps: 2
model:
  class_path: zap.segmentation.models.UNet
  init_args:
    encoder_name: efficientnet-b2
    encoder_depth: 5
    encoder_weights: imagenet
    activation: sigmoid
    num_classes: 1
    loss_fn:
      class_path: torch.nn.BCEWithLogitsLoss
data:
  class_path: zap.segmentation.data_modules.SegmentationDataModule
  init_args:
    data_dir: data/segmentation
    transforms:
      - class_path: torchvision.transforms.ToTensor
      - class_path: torchvision.transforms.Resize
        init_args:
          size:
            - 1024
            - 1024
          antialias: true
    train_split: 0.6
    test_split: 0
    val_split: 0.4
    batch_size: 3
    num_workers: 0
    pin_memory: true
    shuffle: true
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
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

</details>
