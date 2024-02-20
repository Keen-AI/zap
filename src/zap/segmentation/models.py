from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch.nn as nn  # noqa

from ..utils import parse_module_from_string


class DeepLabV3Plus(pl.LightningModule):
    def __init__(self, num_classes, encoder_name, encoder_depth,
                 encoder_weights, activation, loss_fn):
        super().__init__()

        self.model = smp.DeepLabV3Plus(encoder_name=encoder_name,
                                       encoder_depth=encoder_depth,
                                       encoder_weights=encoder_weights,
                                       classes=num_classes,
                                       activation=activation)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def forward(self, x):
        preds = self.model(x)
        return preds

    def training_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)


class UNet(pl.LightningModule):
    def __init__(self, num_classes, encoder_name, encoder_depth,
                 encoder_weights, activation, loss_fn):
        super().__init__()

        self.model = smp.Unet(encoder_name=encoder_name,
                              encoder_depth=encoder_depth,
                              encoder_weights=encoder_weights,
                              classes=num_classes,
                              activation=activation)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def forward(self, x):
        preds = self.model(x)
        return preds

    def training_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        loss = self.loss_fn(output, gt)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
