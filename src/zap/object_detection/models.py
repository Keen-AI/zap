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

import lightning.pytorch as pl
import torch
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetaForObjectDetection, DetaImageProcessor
from transformers.image_transforms import center_to_corners_format


class Deta(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_classes):
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

        # metrics
        self.precision = MeanAveragePrecision(class_metrics=False)

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
            print("Assertion Error!")
            print(pixel_values)
            return {}, {}

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, labels, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict, _, _ = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, _, _ = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict, labels, outputs = self.common_step(batch, batch_idx)

        H, W = labels[0]['orig_size'].tolist()
        labels[0]['labels'] = labels[0].pop('class_labels')

        # format GT bboxes to same format as results (xyxy) and un-normalise
        xyxy_gt_boxes = center_to_corners_format(labels[0]['boxes']).tolist()
        xyxy_scaled_gt_boxes = []
        for b in xyxy_gt_boxes:
            xmin, ymin, xmax, ymax = b

            xmin = xmin * W
            ymin = ymin * H
            xmax = xmax * W
            ymax = ymax * H

            xyxy_scaled_gt_boxes.append([xmin, ymin, xmax, ymax])

        labels[0]['boxes'] = tensor(xyxy_scaled_gt_boxes)

        #  get results from model
        results = self.processor.post_process_object_detection(outputs,
                                                               target_sizes=[(H, W)],
                                                               threshold=0)

        # calc and log precision
        precision = self.precision(results, labels)
        for k, v in precision.items():
            self.log(k, v.item(), on_epoch=True)

        self.log("test_loss", loss)
        for k, v in loss_dict.items():
            self.log("test_" + k, v.item())

        return loss

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
