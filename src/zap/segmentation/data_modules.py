
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
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
        
        self.coco = COCO(self.data_dir.joinpath('coco_labels.json'))
        self.mask_dir = self.data_dir.joinpath('masks')
        
        try:
            os.makedirs(self.mask_dir, exist_ok=False)
            self._generate_masks_binary()
        except FileExistsError as e:
            pass  # TODO: handle
        
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

    def _generate_masks_binary(self):
        cat_ids = self.coco.getCatIds()

        for img in self.coco.imgs.values():
            # get all annotations for this image
            anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(anns_ids)

            # create a mask using label encoding (each pixel is a value representing the class it belongs to)
            mask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
            for a in anns:
                m = self.coco.annToMask(a)
                mask[m > 0] = 255

            filename = Path(img['file_name']).stem + '.png'
            Image.fromarray(mask).save(self.mask_dir.joinpath(filename))

    def _generate_masks_multiclass(self):
        cat_ids = self.coco.getCatIds()

        class_colors = []
        for _ in cat_ids:
            rand_color = random.choices(range(256), k=3)
            class_colors.append(rand_color)

        for img in self.coco.imgs.values():
            # get all annotations for this image
            anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(anns_ids)

            # create a mask using label encoding (each pixel is a value representing the class it belongs to)
            mask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
            for a in anns:
                m = self.coco.annToMask(a)
                color = class_colors[a['category_id']]
                mask[m > 0] = color

            filename = Path(img['file_name']).stem + '.png'
            Image.fromarray(mask).save(filename)
