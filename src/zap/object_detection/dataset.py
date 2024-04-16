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

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


class DETADataset(CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super(DETADataset, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(DETADataset, self).__getitem__(idx)
        file_name = self.coco.loadImgs(idx)[0]["file_name"]

        # preprocess image and target (converting target to DETR format, resizing
        # + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target, file_name


class FasterRCNNDataset(Dataset):
    """Converts a COCO dataset to a format accepted by FasterRCNN"""

    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms

        # load the COCO annotations json
        with open(str(ann_file)) as file_obj:
            self.coco_data = json.load(file_obj)

        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data['annotations']:
            image_id = anno['image_id']
            self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.coco_data['images'])

    def __getitem__(self, index):
        image_data = self.coco_data['images'][index]
        image_id = image_data['id']

        # print('>>>', self.img_folder)
        # print('>>>', image_data['file_name'])

        image_path = self.img_folder / image_data['file_name']
        image = Image.open(image_path).convert('RGB')

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
        }

        for anno in annos:
            coco_bbox = anno['bbox']
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data['boxes'].append([left, top, right, bottom])
            anno_data['labels'].append(anno['category_id'])
            anno_data['area'].append(area)
            anno_data['iscrowd'].append(anno['iscrowd'])

        target = {
            'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64),
            'image_id': torch.tensor([image_id]),  # pylint: disable=not-callable (false alarm)
            'area': torch.as_tensor(anno_data['area'], dtype=torch.float32),
            'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64),
        }

        if self.transforms is not None:
            image = self.transforms(image)
            # target = self.transforms(target)

        return image, target, image_path
