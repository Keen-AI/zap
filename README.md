<div align="center">

<img alt="Lightning" src="logo.png" style="max-width: 100%;">
<br/>

**Zen AI Pipeline**

<small>🚧 IN DEVELOPMENT 🚧</small>

<small>v0.0.4-alpha</small>

</div>

## About

Zap is a lightweight wrapper around Pytorch Lightning and MLFlow. It allows you to quickly train and test models for **classification**, **object detection** and **segmentation** without writing tonnes of boilerplate code.

#### Supported models

| Model       | Type             | Status | Loss              | Import Path                             |
| ----------- | ---------------- | :----: | ----------------- | --------------------------------------- |
| ResNet34    | Classification   |   ✅   | CrossEntropyLoss  | `zap.classification.models.ResNet34`    |
| UNet        | Segmentation     |   ✅   | BCEWithLogitsLoss | `zap.segmentation.models.UNet`          |
| DeepLabV3+  | Segmentation     |   ✅   | BCEWithLogitsLoss | `zap.segmentation.models.DeepLabV3Plus` |
| DETA        | Object Detection |   ✅   | Combination       | `zap.object_detection.models.DETA`      |
| ResNet50    | Classification   |   🚧   | -                 | -                                       |
| ResNet152   | Classification   |   🚧   | -                 | -                                       |
| DenseNet121 | Classification   |   🚧   | -                 | -                                       |
| DenseNet201 | Classification   |   🚧   | -                 | -                                       |
| UNet++      | Segmentation     |   🚧   | -                 | -                                       |
| DeepLabV3   | Segmentation     |   🚧   | -                 | -                                       |
| FasterRCNN  | Object Detection |   🚧   | -                 | -                                       |

> **ℹ️ Coming soon:** dynamic loss functions

#### Things you get for free

These are some of the options that Lightning makes available and we've included in the base config.

| Feature                             | Status | Notes                                          |
| ----------------------------------- | :----: | ---------------------------------------------- |
| Automatic train/test/val splitting  |   ✅   | 70/20/10 (Configurable)                        |
| Automatic Logging                   |   ✅   | Using MLFlow and a database                    |
| Mixed Precision                     |   ✅   | 16-bit mixed                                   |
| Early Stopping                      |   ✅   | Stop training if no improvement after 5 epochs |
| Stochastic Weight Averaging         |   ✅   | Works well with `SGD` and `Adam`               |
| Batch Finder                        |   🚧   | Not robust enough yet                          |
| Learning Rate Finder                |   🚧   | Not working with config-driven optimizers      |
| Gradient Accumulation               |   🚧   | TODO                                           |
| Inference Boilerplate               |   🚧   | TODO                                           |
| Data/result Visualisation           |   🚧   | TODO                                           |
| Detailed val/test metrics and plots |   🚧   | TODO                                           |

## How To Guide

### Installation

```shell
# if using SSH
pip install git+ssh://git@github.com/Keen-AI/zap.git
```

or

```shell
# if using HTTPS
pip install git+https://github.com/Keen-AI/zap.git
```

### Database for logging experiments

MLFlow logs each run, and the hyperparameters and metrics that come with it, to a data store. Whilst [MLFlow supports a number of different types of data stores](https://mlflow.org/docs/latest/tracking/backend-stores.html?highlight=sqlite#supported-store-types) we recommend going with a local or cloud database to start with. We recommend a PostgreSQL database instance, but to get started immediately you can set `ZAP_TRACKING_URI` to `sqlite:///mydb.sqlite` which will create a local SQLite database.

### Environment file

A `.env` file is required to conveniently store some mandatory settings and credentials that you don't want to hardcode.
Create the `.env` file and populate it with these mandatory variables:

```
# the name of your current project/experiment
ZAP_EXPERIMENT_NAME=

# the artifact location is typically a bucket or a local directory; if left blank it will save checkpoints locally
ZAP_ARTIFACT_LOCATION=

# the database connection string
ZAP_TRACKING_URI=
```

### Data and Configuration setup

Depending on which task you need, classification, object detection or segmentation, Zap expects a slightly different data folder structure.

Here are the guides for each task:

- [Classification](docs/classification_guide.md)
- [Object Detection](docs/object_detection_guide.md)
- [Segmentation](docs/segmentation_guide.md)

Once you've finished the guide you need, return to this step.

## Running

In order to run things let's create a `main.py` file.

```python
from zap import Zap

z = Zap()

if __name__ == '__main__':
    z.fit()              # training and validation
    z.test()             # testing using the best checkpoint
```

Now we can call with file with an additional argument that specifies our config file:

```
python main.py -c config.yaml
```

### Saving checkpoints

Zap will automatically save the checkpoints of each epoch in a folder called `mlruns` that will appear when you first kick off a training job.

It will only save the **best** checkpoint, which is determined by having the **lowest validation loss**.

> ℹ️ Currently the MLFlow logger creates a numbered folder inside `mlruns` for each experiment; inside the numbered folder it creates another folder for the current run, named with a long GUID. Not sure why it uses numbers and GUIDs instead of the experiment name and the human-friendly run IDs it displays in the UI.

> ⚠️ Currently checkpoints are only uploaded to a bucket if the training run finishes "naturally", i.e. if the early stopping kicks. If you cancel a run or if the trainer reaches the maximum number of epochs (100), the checkpoints are only stored locally

### Testing separately

If you run `z.fit()` and `z.test()` in the same run, the test command will automatically reference the best checkpoint (currently Zap determines "best" as the weights that resulted in the lowest validation loss). However, if you run `z.test()` separately, you will need to specify the checkpoint path:

```python
z.test(ckpt_path='mlruns/.../...')
```

## Viewing Experiment Results

The results of your experiment will be logged in the database and any checkpoints (and other artifacts) will be saved locally (and remotely, if setup that way). You can start the server locally to browse the data from your runs:

Replace the variables with their actual values and run:

```
mlflow server \
  --backend-store-uri ${ZAP_TRACKING_URI}
  --artifacts-destination ${ZAP_ARTIFACT_LOCATION}
```

You can set up MLFlow to [run on a VM](https://mlflow.org/docs/latest/tracking/server.html#secure-tracking-server) to make it accessible to others, though if you're working in production you'll likely want to setup [authentication](https://mlflow.org/docs/latest/auth/index.html?highlight=authentication).

## Inference

Once you've trained a model and want to use it for inference, you can do the following:

```python
from zap import Zap

z = Zap()

if __name__ == '__main__':
    predictions = z.predict(ckpt_path='path_to_your_ckpt')

    for p in predictions:
        pass  # do whatever you need to do
```

You still need to call your script with the config file:

```
python infer.py -c config.yaml
```
