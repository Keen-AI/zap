
from typing import Any, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, images, labels, label_map, transform=None) -> None:
        self.images = images
        self.labels = labels
        self.label_map = label_map
        self.transform = transform
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.images[index]
        img = Image.open(img)
        # img = Image.fromarray(img.numpy(), mode="L")
        
        label = self.labels.loc[self.images[index].name, 'label']
        label = self.label_map[label]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
