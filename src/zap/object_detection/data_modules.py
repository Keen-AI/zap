


from .. import ZapDataModule
from .dataset import ObjectDetectionDataset

from torch.utils.data import DataLoader
from transformers import DetaImageProcessor


class ObjectDetectionDataModule(ZapDataModule):
    def __init__(self, data_dir, size, batch_size=1, num_workers=0, pin_memory=True, shuffle=True):
        super().__init__()
        self.processor = DetaImageProcessor.from_pretrained(
            "jozhang97/deta-resnet-50",
            size={
                "shortest_edge": size[0],
                "longest_edge": size[1]})
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.train_dataset = ObjectDetectionDataset(
                img_folder=data_dir+"/train",
                processor=self.processor)
        self.val_dataset = ObjectDetectionDataset(
                img_folder=data_dir+"/val",
                processor=self.processor, train=False)
        self.test_dataset = ObjectDetectionDataset(
                img_folder=data_dir+"/val",
                processor=self.processor,
                train=False)
