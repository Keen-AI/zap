
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import MulticlassPrecision


# TODO: can we make this dynamic in terms of which ResNet is used like we did in kai-classification?
class ResNet34(pl.LightningModule):
    def __init__(self, num_classes, bias):
        super().__init__()

        self.model = models.resnet.resnet34(weights="DEFAULT")
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=bias)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.multiclass_precision = MulticlassPrecision(num_classes=num_classes)

    def forward(self, x):
        pred = self.model(x)
        probabilities = nn.functional.softmax(pred, dim=1)
        return probabilities

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self.model(img)

        loss = self.loss_fn(output, label)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self.model(img)
        loss = self.loss_fn(output, label)
        self.log('val_loss', loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self.model(img)
        loss = self.loss_fn(output, label)
        self.log('test_loss', loss)

        mcp = self.multiclass_precision(output, label, average=None)
        self.log('precision', mcp)


