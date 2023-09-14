import os
from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp


class UNet(pl.LightningModule):
    def __init__(self, num_classes, encoder_name, encoder_depth, encoder_weights, activation, loss_fn):
        super().__init__()

        self.model = smp.Unet(encoder_name=encoder_name, 
                              encoder_depth=encoder_depth,
                              encoder_weights=encoder_weights, 
                              classes=num_classes, 
                              activation=activation)
    
        self.loss_fn = loss_fn

        self.automatic_optimization = False  # required for this model
        self.checkpoint_save_dir = None  # required when auto optim is False
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()
    
    def forward(self, x):
        pred = self.model(x)
        return pred

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
        self.log('test_loss', loss)

    def on_fit_start(self) -> None:
        """Create the directory to save checkpoints"""
        self.checkpoint_save_dir =  os.path.join(self.logger.save_dir, 
                                                 self.logger._experiment_id, 
                                                 self.logger._run_id, 
                                                 'model', 'checkpoints')
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        return super().on_fit_start()

    def on_validation_epoch_end(self) -> None:
        """Save the checkpoint on end of each epoch"""
        filename = f'ckpt_{self.trainer.current_epoch}.pt'
        self.trainer.save_checkpoint(os.path.join(self.checkpoint_save_dir, filename))