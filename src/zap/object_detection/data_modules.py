import json
import os
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from pycocotools.coco import COCO
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

from ..utils import get_label_map
from .dataset import ObjectDetectionDataset


class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transforms, train_split=0.7, test_split=0.2, val_split=0.1, batch_size=1, num_workers=0, pin_memory=True, shuffle=True):
        super().__init__()

        self.data_dir = Path(data_dir)
        
        # TODO: do we even need the images object if we have the COCO obj?
        self.image_dir = self.data_dir.joinpath('images')
        self.images = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg'))
        
        self.coco = COCO(self.data_dir.joinpath('labels.json'))
        
        self.transforms = Compose(transforms)

        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.shuffle=shuffle

        # TODO: add image checks like in classif case

        dataset = ObjectDetectionDataset(self.images, self.coco)
        generator = Generator()  # TODO: look into using sklearn.model_selection.train_test_split instead
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(dataset, [train_split, test_split, val_split], generator)
        
        
        self.predict_dir = Path(data_dir, 'predict', 'images')
        prediction_images = list(self.predict_dir.glob('*.png')) + list(self.predict_dir.glob('*.jpg'))
        # self.predict_dataset = InferenceDatasetNew(prediction_images, transform=self.transforms)  # TODO: <<

        self.save_hyperparameters()

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
