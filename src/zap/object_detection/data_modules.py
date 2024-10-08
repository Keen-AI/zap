"""
Adapted from: https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA

MIT License

Copyright (c) 2021 NielsRogge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from pathlib import Path

from torch import Generator
from torch.utils.data import random_split
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import Compose
from transformers import DetaImageProcessor

from .. import InferenceDataset, ZapDataModule
from ..utils import parse_module_from_string
from .dataset import DETADataset, FasterRCNNDataset


class DETADataModule(ZapDataModule):
    def __init__(self, data_dir, size, batch_size=1, num_workers=0, pin_memory=True, transforms=None,
                 shuffle=True, train_split=0.7, test_split=0.2, val_split=0.1, converter=None):

        if converter:
            converter_fn = parse_module_from_string(converter)
            converter_fn(data_dir)

        self.processor = DetaImageProcessor.from_pretrained(
            "jozhang97/deta-resnet-50",  # TODO: make this config driven
            size={
                "shortest_edge": size[0],
                "longest_edge": size[1]})

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        #  TODO: investigate how to use these transforms when training DETA
        #  Currently only used/required for inference
        self.transforms = Compose(transforms) if transforms else None

        dataset = DETADataset(
            img_folder=self.data_dir / 'images',
            ann_file=self.data_dir / 'labels.json',
            processor=self.processor)

        self.label_map = dataset.label_map

        generator = Generator().manual_seed(42)
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_split, test_split, val_split], generator)

        super().__init__()
        self.save_hyperparameters()

    def predict_collate_fn(self, batch):
        pixel_values = []
        for img in batch:
            encoding = self.processor(images=img, return_tensors="pt")
            pixel_values.append(encoding["pixel_values"].squeeze())

        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        return encoding['pixel_values']

    def collate_fn(self, batch):
        file_names = [item[2] for item in batch]
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]

        batch = {}
        batch['file_names'] = file_names
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels

        return batch


class FasterRCNNDataModule(ZapDataModule):
    def __init__(self, data_dir, batch_size=1, num_workers=0, pin_memory=True, transforms=None,
                 shuffle=True, train_split=0.7, test_split=0.2, val_split=0.1):

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        # NOTE: default transforms are not compatible with ToTensor; should warn user?
        # NOTE: from testing on generated dataset, anything other than the default transforms yields terrible results
        default_transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
        self.transforms = Compose([default_transforms, *transforms])

        dataset = FasterRCNNDataset(
            img_folder=self.data_dir / 'images',
            ann_file=self.data_dir / 'labels.json',
            transforms=self.transforms)

        self.label_map = dataset.label_map

        generator = Generator().manual_seed(42)
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_split, test_split, val_split], generator)

        super().__init__()
        self.save_hyperparameters()

    def collate_fn(self, batch):
        """return tuple data"""
        return tuple(zip(*batch))
