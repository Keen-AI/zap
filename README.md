<div align="center">

<img alt="Lightning" src="logo.png" style="max-width: 100%;">
<br/>

**Zen AI Pipeline**

<small>🚧 IN DEVELOPMENT 🚧</small>

<small>v0.0.3-alpha</small>

</div>

## About

Zap is a lightweight wrapper around Pytorch Lightning and MLFlow. It allows you to quickly train and test models for **classification**, **object detection** and **segmentation** without writing tonnes of boilerplate code.

#### Supported models

| Model         | Type                    | Status | Import Path                          |
| ------------- | ----------------------- | :----: | ------------------------------------ |
| ResNet34      | Classification          |   ✅   | `zap.classification.models.ResNet34` |
| ResNet50      | Classification          |   🚧   | N/A                                  |
| ResNet152     | Classification          |   🚧   | N/A                                  |
| DenseNet121   | Classification          |   🚧   | N/A                                  |
| DenseNet201   | Classification          |   🚧   | N/A                                  |
| UNet          | Segmentation            |   ✅   | `zap.segmentation.models.UNet`       |
| UNet++        | Segmentation            |   🚧   | N/A                                  |
| DeepLabV3     | Segmentation            |   🚧   | N/A                                  |
| DeepLabV3+    | Segmentation            |   🚧   | N/A                                  |
| FasterRCNN    | Object Detection        |   🚧   | N/A                                  |
| YOLO          | Multi                   |   🚧   | N/A                                  |
| DETA          | Object Detection        |   🚧   | N/A                                  |
| RTMDet        | RT Object Detection     |   🚧   | N/A                                  |
| RT-DETR       | RT Object Detection     |   🚧   | N/A                                  |
| GroundingDino | 0-Shot Object Detection |   🚧   | N/A                                  |

#### Things you get for free

| Feature                     | Status | Notes                                     |
| --------------------------- | :----: | ----------------------------------------- |
| Automatic Logging           |   ✅   |                                           |
| Mixed Precision             |   ✅   | 16-bit mixed                              |
| Early Stopping              |   ✅   |                                           |
| Stochastic Weight Averaging |   ✅   | Works well with `SGD` and `Adam`          |
| Batch Finder                |   🚧   | Not robust enough yet                     |
| Learning Rate Finder        |   🚧   | Not working with config-driven optimizers |
| Gradient Accumulation       |   🚧   | Advanced feature, coming soon             |

> **ℹ️ NOTE:** you can add any of the not-implemented features to your own config

## How To Guide

#### Installation

```shell
# if using SSH
pip install git+ssh://git@github.com/Keen-AI/zap.git

# if using HTTPS
pip install git+https://github.com/Keen-AI/zap.git
```

### GCP Setup

Zap uses MLFlow to track runs and store their artifacts (e.g: checkpoints). MLFlow stores metadata for each run in a Postgres database, and artifacts in a Google Storage Bucket.

This guide assumes you've installed the GCP CLI.

1. Create a `.env` file and populate it with the master credentials:

   ```
   ZAP_DB_USER=
   ZAP_DB_PWD=
   ZAP_DB_HOST=
   ZAP_DB_PORT=
   ZAP_DB_NAME=

   ZAP_BUCKET=
   ```

2. Set your GCP project to `internal`: `gcloud config set project <PROJECT_ID>`

3. Authenticate the Google Cloud SDK: `gcloud auth application-default login`

### Project Setup

Using Zap is super simple. There are 2 key steps:

- Create the required config for your problem
- Create a Python script to call it

#### Project Structure

```
my_project/
├─ data/
├─ main.py
├─ config.yaml

```

Your `data` folder will look differently depending on the type of problem you have.

> ⚠️ TODO: add documentation on data setup for each problem (classification, segmentation, etc)

#### Creating the config

Let's start with the config. Zap has already done lot of the base configuration for you. We just need to tell it 3 things:

- the model we want to use
- the data we want to load
- the optimizer we want to use

Create a `.yaml` file for your configuration. Let's define the model first. For a list of available models, see the Available Models section.

```yaml
model:
  class_path: zap.classification.models.ResNet34
  init_args:
    num_classes: 12
    bias: true
```

Now, in the same file, let's define the data source. We're also going to define:

- transforms (in this example, `ToTensor` and `Resize` to `32x32`)
- our train, test, val splits
- our batch size (and a few other typical Pytorch settings)

```yaml
data:
  class_path: zap.classification.data_modules.ClassificationDataModule
  init_args:
    data_dir: <YOUR_DATA_DIR> # see project structure section
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
```

And finally let's define our optimizer. In this example we'll use `Adam` with a learning rate of 0.001. For more optimizers you can explore the `torch.optim` module (or Pytorch docs), **but keep in mind that the Stochastic Weight Averaging technique does not support all optimizers.**

```yaml
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

#### Creating the script

This is the easy part. At the bare minimum, your script needs to:

- create an instance of `Zap` and set the `experiment_name` param
- call `fit`, `test`, `predict` based on your needs

```python
from zap import Zap

z = Zap(experiment_name='my_experiment', env_location='.env')

if __name__ == '__main__':
    z.fit()
    z.test()
    preds = z.predict()
```

#### Running

**First make sure you're running the Cloud SQL Proxy!**

We can call our `main.py` script and give it our `config.yaml` from the command line as follows:

```
python main.py -c config.yaml
```

## Viewing Results

The results of your experiment will be logged in a central database and any checkpoints (and other artifacts) will be saved in a central bucket. We don't have a server running the MLFlow app at the moment but you can easily run it locally whilst still connected to the central repositories.

**First make sure you're running the Cloud SQL Proxy!**

Then replace the environment variables with their actual values and run:

```
mlflow server \
  --backend-store-uri postgresql://${ZAP_DB_USER}:${ZAP_DB_PWD}@${ZAP_DB_HOST}:${ZAP_DB_PORT}/${ZAP_DB_NAME}
  --artifacts-destination gs://<ZAP_BUCKET>/<YOUR_EXPERIMENT>
```

## Overriding defaults

TODO

## Zap development

TODO
