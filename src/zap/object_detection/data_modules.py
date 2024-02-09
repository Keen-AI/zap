from .. import ZapDataModule, InferenceDataset
from .dataset import ObjectDetectionDataset
from ..utils import parse_loss_fn_module
from transformers import DetaImageProcessor
from torch.utils.data import random_split
from pathlib import Path
from torch import Generator


class ObjectDetectionDataModule(ZapDataModule):
    def __init__(self, data_dir, size, batch_size=1, num_workers=0, pin_memory=True,
                 shuffle=True, collate_fn=None, train_split=0.7, test_split=0.2, val_split=0.1, converter=None):
        super().__init__()

        if converter:
            converter_fn = parse_loss_fn_module(converter)
            converter_fn(data_dir)

        self.processor = DetaImageProcessor.from_pretrained(
            "jozhang97/deta-resnet-50",
            size={
                "shortest_edge": size[0],
                "longest_edge": size[1]})

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.collate_fn = parse_loss_fn_module(collate_fn)

        dataset = ObjectDetectionDataset(
            img_folder=data_dir,
            processor=self.processor)

        generator = Generator()
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_split, test_split, val_split], generator)

        self.predict_dir = Path(data_dir, 'predict', 'images')
        prediction_images = list(self.predict_dir.glob(
            '*.png')) + list(self.predict_dir.glob('*.jpg'))
        self.predict_dataset = InferenceDataset(
            prediction_images)

        self.save_hyperparameters()
