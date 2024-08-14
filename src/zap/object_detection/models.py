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


import torch
from torch import tensor
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import DetaForObjectDetection, DetaImageProcessor
from transformers.image_transforms import center_to_corners_format

from .. import ZapModel


class Deta(ZapModel):
    def __init__(self, lr, lr_backbone, weight_decay, num_classes):
        self.task = 'object_detection'
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.pretrained_model = "jozhang97/deta-resnet-50"  # TODO: make this config driven
        self.model = DetaForObjectDetection.from_pretrained(self.pretrained_model,
                                                            num_labels=num_classes,
                                                            auxiliary_loss=True,
                                                            ignore_mismatched_sizes=True)

        self.processor = DetaImageProcessor.from_pretrained(self.pretrained_model,
                                                            size={"shortest_edge": 432,  # TODO: needs to match config
                                                                  "longest_edge": 768})

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)
        return optimizer

    def forward(self, pixel_values, pixel_mask=None):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        try:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        except AssertionError:  # TODO: handle this properly
            print(f"Assertion Error at {batch['file_names']}")
            return {}, {}, {}, {}

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, labels, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict, _, _ = self.common_step(batch, batch_idx)
        if not loss:
            return None

        self.log("train_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, _, _ = self.common_step(batch, batch_idx)
        if not loss:
            return None

        self.log("val_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict, labels, outputs = self.common_step(batch, batch_idx)
        if not loss:
            return None

        target_sizes = []
        for n in range(len(labels)):
            H, W = labels[n]['orig_size'].tolist()

            # rename labels key
            labels[n]['labels'] = labels[n].pop('class_labels')

            # format GT bboxes to same format as results (xyxy) and un-normalise
            xyxy_gt_boxes = center_to_corners_format(labels[n]['boxes']).tolist()
            xyxy_scaled_gt_boxes = []
            for b in xyxy_gt_boxes:
                xmin, ymin, xmax, ymax = b

                xmin = xmin * W
                ymin = ymin * H
                xmax = xmax * W
                ymax = ymax * H

                xyxy_scaled_gt_boxes.append([xmin, ymin, xmax, ymax])

            labels[n]['boxes'] = tensor(xyxy_scaled_gt_boxes)
            target_sizes.append((H, W))

        #  get results from model
        results = self.processor.post_process_object_detection(outputs,
                                                               target_sizes=target_sizes,
                                                               threshold=0)
        self.mAP(results, labels)


class FasterRCNN(ZapModel):
    def __init__(self, num_classes):
        self.task = 'object_detection'
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

        # Replace the pre-trained head with a new head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.save_hyperparameters()

    def on_validation_model_eval(self):
        self.trainer.model.train()  # leave model in training mode for validation step so we can get val losses

    def configure_optimizers(self):
        return super().configure_optimizers()

    def forward(self, batch):
        preds = self.model(batch)
        return preds

    def common_step(self, batch):
        images, targets, filenames = batch
        images = list(images)
        targets = list(targets)
        filenames = list(filenames)

        return images, targets, filenames

    def training_step(self, batch, batch_idx):
        images, targets, _ = self.common_step(batch)

        loss_dict = self.model(images, targets)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_epoch=True)

        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = self.common_step(batch)

        loss_dict = self.model(images, targets)
        for k, v in loss_dict.items():
            self.log("val_" + k, v.item(), on_epoch=True)

        loss = sum(loss for loss in loss_dict.values())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets, _ = self.common_step(batch)

        results = self.model(images)
        self.mAP(results, targets)
