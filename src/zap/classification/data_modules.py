import os
from pathlib import Path

import pandas as pd
from torch import Generator
from torch.utils.data import random_split
from torchvision.transforms import Compose

from .. import ZapDataModule
from .dataset import ClassificationDataset


class ClassificationDataModule(ZapDataModule):
    def __init__(self, data_dir, label_map, transforms, train_split=0.7, test_split=0.2, val_split=0.1,
                 batch_size=1, num_workers=0, pin_memory=True, shuffle=True):

        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir.joinpath('images')
        self.images = list(self.image_dir.glob('*.*'))
        self.train_split = train_split
        self.test_split = test_split
        self.val_split = val_split

        self.label_map = label_map
        self.label_map_reversed = {v: k for k, v in label_map.items()}

        self.transforms = Compose(transforms)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.collate_fn = None

        super().__init__()  # important to initialise here
        self.save_hyperparameters()

    def setup(self, stage):
        if stage == 'predict':
            return

        self.labels_df = pd.read_csv(self.data_dir.joinpath('labels.csv'))
        self.labels_df['image'] = self.labels_df['image'].apply(
            lambda x: os.path.basename(x))
        self.labels_df = self.labels_df.drop_duplicates()  # TODO: raise warnings
        self.labels_df = self.labels_df.dropna()
        self.labels_df = self.labels_df.set_index('image')
        self.labels_df = self.labels_df[['label']]

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
            dataset, [self.train_split, self.test_split, self.val_split], generator)
