
from pathlib import Path

from torch import Generator
from torch.utils.data import random_split
from torchvision.transforms import Compose

from .. import InferenceDataset, ZapDataModule
from .dataset import SegmentationDataset


class SegmentationDataModule(ZapDataModule):
    def __init__(self, data_dir, transforms, train_split=0.7, test_split=0.2, val_split=0.1, 
                 batch_size=1, num_workers=0, pin_memory=True, shuffle=True):
        super().__init__()

        self.data_dir = Path(data_dir)
        
        self.image_dir = self.data_dir.joinpath('images')

        # TODO: fix allowed extensions
        self.images = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg'))
        
        self.mask_dir = self.data_dir.joinpath('masks')
        self.masks = list(self.mask_dir.glob('*.png'))
        self.transforms = Compose(transforms)

        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.shuffle=shuffle

        dataset = SegmentationDataset(self.images, self.masks, transform=self.transforms)
        generator = Generator()  # TODO: look into using sklearn.model_selection.train_test_split instead
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(dataset, [train_split, test_split, val_split], generator)
        
        # TODO: cleanup
        self.predict_dir = Path(data_dir, 'predict', 'images')
        prediction_images = list(self.predict_dir.glob('*.png')) + list(self.predict_dir.glob('*.jpg'))
        self.predict_dataset = InferenceDataset(prediction_images, transform=self.transforms)

        self.save_hyperparameters()