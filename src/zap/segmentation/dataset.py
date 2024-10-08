from typing import Any, Tuple

from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transforms=None) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transform
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')

        if self.transforms is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
