import os

from dotenv import load_dotenv
from lightning.pytorch.loggers import MLFlowLogger

from .formatter import (format_lightning_warnings_and_logs,
                        supress_pydantic_warnings)

supress_pydantic_warnings()  # noqa

from typing import Any  # noqa

from lightning.pytorch import LightningDataModule, LightningModule  # noqa
from lightning.pytorch.cli import LightningCLI  # noqa
from PIL import Image  # noqa
from torch.utils.data import DataLoader, Dataset  # noqa
from torchmetrics.detection import MeanAveragePrecision  # noqa

format_lightning_warnings_and_logs()


class Zap():
    def __init__(self) -> None:
        load_dotenv('.env')

        package_path = os.path.dirname(os.path.realpath(__file__))
        base_config_path = os.path.join(package_path, 'base.yaml')

        self.cli = LightningCLI(save_config_kwargs={"overwrite": True}, save_config_callback=None, run=False,
                                parser_kwargs={"parser_mode": "omegaconf",
                                               "default_config_files": [base_config_path]})
        self.config = self.cli.config.as_dict()

        self.cli.trainer.logger.log_hyperparams({'optimizer': self.config.get('optimizer')})

    def fit(self):
        self.cli.trainer.fit(self.cli.model, self.cli.datamodule)

    def test(self, ckpt_path="best"):
        self.cli.trainer.test(self.cli.model, self.cli.datamodule, ckpt_path=ckpt_path)

    def predict(self, ckpt_path="last"):
        preds = self.cli.trainer.predict(
            self.cli.model,
            self.cli.datamodule,
            return_predictions=True,
            ckpt_path=ckpt_path)
        return preds


# TODO: test compatibility with classification and segmentation models
class ZapModel(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.task = getattr(self, 'task', None)
        self.mAP = None
        if self.task == 'object_detection':
            self.mAP = MeanAveragePrecision(class_metrics=True)

    def on_fit_end(self) -> None:
        self.logger.experiment.log_param(self.logger.run_id, 'train_set', len(self.trainer.datamodule.train_dataset))
        self.logger.experiment.log_param(self.logger.run_id, 'test_set', len(self.trainer.datamodule.test_dataset))
        self.logger.experiment.log_param(self.logger.run_id, 'val_set', len(self.trainer.datamodule.val_dataset))

    def on_test_epoch_start(self) -> None:
        self.label_map = self.trainer.datamodule.label_map

    def on_test_end(self) -> None:
        if self.mAP:
            mAP = self.mAP.compute()
            self.log_precision(mAP)

    def log_precision(self, precision):
        classes = precision.pop('classes', None)  # get the classes but don't log them

        for k, v in precision.items():
            print('>>>>>', k, v)
            k = k.replace('map', 'mAP').replace('mar', 'mAR')
            class_values = v.tolist()

            # handling single class vs multiclass
            if k in ('mAP_per_class', 'mAR_100_per_class'):
                for pair in zip(classes, class_values):  # log each class metric separately
                    k = k.replace('_per_class', '')
                    self.label_map[int(pair[0])]
                    self.logger.experiment.log_metric(
                        self.logger.run_id, f'{k}_{self.label_map[int(pair[0])]}', pair[1])
            else:
                self.logger.experiment.log_metric(self.logger.run_id, k, class_values)


class ZapDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

        # we get the data_dir from the data module class that inherits this class
        self.data_dir = getattr(self, 'data_dir', None)
        if not self.data_dir:
            raise AttributeError('Missing data directory')

        self.predict_dir = self.data_dir / 'predict' / 'images'
        prediction_images = list(self.predict_dir.glob('*.png')) + list(self.predict_dir.glob('*.jpg'))
        self.predict_dataset = InferenceDataset(prediction_images, transforms=self.transforms)

        # batch the images and expose for convenience during inference
        self.prediction_images = []
        for i in range(0, len(prediction_images), self.batch_size):
            self.prediction_images.append(tuple(prediction_images[i:i + self.batch_size]))

    def prepare_data(self, bucket=None):
        # NOTE: do not assign state here (e.g: self.x = 123)
        # therefore the data just needs to be saved locally here
        if bucket:
            pass  # TODO: download data from bucket/database

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=self.shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        collate_fn = self.predict_collate_fn if hasattr(self, 'predict_collate_fn') else None
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False, collate_fn=collate_fn)


class InferenceDataset(Dataset):
    def __init__(self, images, transforms=None) -> None:
        self.images = images
        self.transforms = transforms
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.images[index]).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        # TODO: test compatibility with all models
        return img
