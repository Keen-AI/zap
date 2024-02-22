
# Adapted from: https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA

import torchvision


class ObjectDetectionDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super(ObjectDetectionDataset, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(ObjectDetectionDataset, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing
        # + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target
