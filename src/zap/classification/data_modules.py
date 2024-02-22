import os
from pathlib import Path

import pandas as pd
from torch import Generator
from torch.utils.data import random_split
from torchvision.transforms import Compose

from .. import InferenceDataset, ZapDataModule
from ..utils import get_label_map
from .dataset import ClassificationDataset


class ClassificationDataModule(ZapDataModule):
    def __init__(self, data_dir, transforms, train_split=0.7, test_split=0.2, val_split=0.1,
                 batch_size=1, num_workers=0, pin_memory=True, shuffle=True, collate_fn=None):
        super().__init__()

        self.data_dir = Path(data_dir)

        self.image_dir = self.data_dir.joinpath('images')
        self.images = list(self.image_dir.glob('*.*'))

        self.label_map, self.label_map_reversed = get_label_map(
            self.data_dir.joinpath('label_map.json'))
        self.labels_df = pd.read_csv(self.data_dir.joinpath('labels.csv'))
        self.labels_df['image'] = self.labels_df['image'].apply(
            lambda x: os.path.basename(x))
        self.labels_df = self.labels_df.drop_duplicates()  # TODO: raise warnings
        self.labels_df = self.labels_df.dropna()
        self.labels_df = self.labels_df.set_index('image')
        self.labels_df = self.labels_df[['label']]

        self.transforms = Compose(transforms)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        filtered_images = []
        for i in self.images:
            if os.path.basename(i) in self.labels_df.index:
                filtered_images.append(i)
            else:
                pass  # TODO: raise warning

        dataset = ClassificationDataset(
            filtered_images, self.labels_df, self.label_map, self.transforms)
        #  TODO: look into using sklearn.model_selection.train_test_split instead
        generator = Generator()
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_split, test_split, val_split], generator)

        self.predict_dir = Path(data_dir, 'predict', 'images')
        prediction_images = list(self.predict_dir.glob(
            '*.png')) + list(self.predict_dir.glob('*.jpg'))
        self.predict_dataset = InferenceDataset(
            prediction_images, transform=self.transforms)

        self.save_hyperparameters()
