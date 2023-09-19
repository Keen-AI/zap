
from typing import Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Zap():
    def __init__(self) -> None:
        self.zap = LightningCLI(save_config_kwargs={"overwrite": True}, run=False)
        self.config = self.zap.config.as_dict()
        self.zap.trainer.logger.log_hyperparams({'optimizer': self.config['optimizer']})
    
    def fit(self):
        self.zap.trainer.fit(self.zap.model, self.zap.datamodule)

    def test(self):
        self.zap.trainer.test(self.zap.model, self.zap.datamodule, ckpt_path="best")

    def predict(self):
        preds = self.zap.trainer.predict(self.zap.model, self.zap.datamodule, return_predictions=True, ckpt_path="last")
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