import os
from pprint import pprint

from dotenv import load_dotenv

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
        self.cli.trainer.logger.log_hyperparams({'train_set': len(self.cli.datamodule.train_dataset)})
        self.cli.trainer.logger.log_hyperparams({'test_set': len(self.cli.datamodule.test_dataset)})
        self.cli.trainer.logger.log_hyperparams({'val_set': len(self.cli.datamodule.val_dataset)})

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
        self.mAP = MeanAveragePrecision(class_metrics=True)

    def on_test_epoch_start(self) -> None:
        self.label_map = self.trainer.datamodule.label_map

    def on_test_end(self) -> None:
        mAP = self.mAP.compute()
        pprint(mAP)  # TODO: use the CLI logger?
        self.log_precision(mAP)

    def log_precision(self, precision):
        classes = precision.pop('classes', None)  # get the classes but don't log them

        for k, v in precision.items():
            # handling single class vs multiclass
            class_values = v.tolist()
            if not isinstance(class_values, list):
                self.logger.experiment.log_metric(self.logger.run_id, k, class_values)
            else:
                for pair in zip(classes, class_values):  # log each class metric separately
                    k = k.replace('_per_class', '_class')
                    self.logger.experiment.log_metric(self.logger.run_id, f'{k}_{self.label_map[pair[0]]}', pair[1])


class ZapDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def prepare_data(self, bucket=None):
        # NOTE: do not assign state here (e.g: self.x = 123)
        # therefore the data just needs to be saved locally here
        if bucket:
            pass  # TODO: download data from bucket/database

    def setup(self, stage):
        # NOTE: this is called for each of trainer.train|test|validate
        # as such I don't see why we need to split the datasets here and not in the __init__
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
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False, collate_fn=self.collate_fn)


class InferenceDataset(Dataset):
    def __init__(self, images, transform=None) -> None:
        self.images = images
        self.transform = transform
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.images[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # TODO: test compatibility with all models
        return img, self.images[index]
