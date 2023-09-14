import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ObjectDetectionDataset(Dataset):
    def __init__(self, images, coco, transforms=None):
        self.images = images
        self.coco = coco
        self.transform = transforms  # TODO: rename
        super().__init__()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img)
        
        bbox = self.labels['annotations']
        label = self.labels.loc[self.images[index].name, 'label']

        # NOTE: do we need to do this? what about the precision?
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        target = {}
        target["boxes"] = bbox
        target["labels"] = label
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target





class CocoDetection(Dataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        data_dir: str,
        : str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


