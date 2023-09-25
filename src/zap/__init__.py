import os

from .formatter import (format_lightning_warnings_and_logs,
                        supress_pydantic_warnings)

supress_pydantic_warnings()  # noqa

from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from PIL import Image
from torch.utils.data import DataLoader, Dataset

format_lightning_warnings_and_logs()

class Zap():
    def __init__(self, experiment_name) -> None:
        os.environ['ZAP_ENV_NAME'] = experiment_name
        self.cli = LightningCLI(save_config_kwargs={"overwrite": True}, run=False, 
                                parser_kwargs={"parser_mode": "omegaconf", 
                                               "default_config_files": ['base.yaml']})
        
        self.config = self.cli.config.as_dict()

        self.cli.trainer.logger.log_hyperparams({'optimizer': self.config['optimizer']})
        self.cli.trainer.logger.log_hyperparams({'train_set': len(self.cli.datamodule.train_dataset)})
        self.cli.trainer.logger.log_hyperparams({'test_set': len(self.cli.datamodule.test_dataset)})
        self.cli.trainer.logger.log_hyperparams({'val_set': len(self.cli.datamodule.val_dataset)})
    
    def fit(self):
        self.cli.trainer.fit(self.cli.model, self.cli.datamodule)

    def test(self):
        self.cli.trainer.test(self.cli.model, self.cli.datamodule, ckpt_path="best")

    def predict(self):
        preds = self.cli.trainer.predict(self.cli.model, self.cli.datamodule, return_predictions=True, ckpt_path="last")
        return preds
    
    
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
                          shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          shuffle=False)




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

        return img