
from typing import Any

import lightning.pytorch as pl
import torch
import torchvision.models as models


class MyResnetModel(pl.LightningModule):
    def __init__(self, lr, momentum, num_classes, bias):
        super().__init__()

        self.model = models.resnet.resnet34(weights="DEFAULT")  # TODO: is DEFAULT still advised?
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=bias)

        self.loss_fn = torch.nn.CrossEntropyLoss()  # TODO: make configurable
        self.lr = lr
        self.momentum = momentum

        self.save_hyperparameters()
        

    def forward(self, x):
        pass

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer

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
        self.log('test_loss', loss, on_epoch=True, on_step=True)