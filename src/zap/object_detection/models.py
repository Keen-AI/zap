
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision.models.detection as models
from torchmetrics.detection import IntersectionOverUnion


class ResNet50_FasterRCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.model = model = models.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new head
        self.loss_fn = IntersectionOverUnion()
        self.model.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        self.save_hyperparameters()

    def forward(self, x):
        pred = self.model(x)
        # probabilities = nn.functional.softmax(pred, dim=1)
        return pred
    
    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def training_step(self, batch, batch_idx):
        img, targets = batch
        output = self.model(img)
        # TODO: figure out data structure
        loss = self.loss_fn(output, targets)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self.model(img)
        # loss = self.loss_fn(output, label)
        # self.log('val_loss', loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self.model(img)
        # loss = self.loss_fn(output, label)
        # self.log('test_loss', loss)