from pathlib import Path

from torch import Generator
from torch.utils.data import random_split
from torchvision.transforms import Compose
from transformers import DetaImageProcessor

from .. import InferenceDataset, ZapDataModule
from ..utils import parse_module_from_string
from .dataset import ObjectDetectionDataset


class ObjectDetectionDataModule(ZapDataModule):
    def __init__(self, data_dir, size, batch_size=1, num_workers=0, pin_memory=True, transforms=None,
                 shuffle=True, train_split=0.7, test_split=0.2, val_split=0.1, converter=None):
        super().__init__()

        if converter:
            converter_fn = parse_module_from_string(converter)
            converter_fn(data_dir)

        self.processor = DetaImageProcessor.from_pretrained(
            "jozhang97/deta-resnet-50",  # TODO: make this config driven
            size={
                "shortest_edge": size[0],
                "longest_edge": size[1]})

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        #  TODO: investigate how to use these transforms when training DETA
        #  Currently only used/required for inference
        self.transforms = Compose(transforms) if transforms else None

        dataset = ObjectDetectionDataset(
            img_folder=data_dir,
            processor=self.processor)

        generator = Generator()
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_split, test_split, val_split], generator)

        self.predict_dir = Path(data_dir, 'predict', 'images')
        prediction_images = list(self.predict_dir.glob('*.png')) + list(self.predict_dir.glob('*.jpg'))
        # TODO: rename transforms to be consistent
        self.predict_dataset = InferenceDataset(prediction_images, transform=self.transforms)

        self.save_hyperparameters()

    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]

        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels

        return batch
