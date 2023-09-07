
from lightning.pytorch.cli import LightningCLI


class Zap():
    def __init__(self) -> None:
        self.zap = LightningCLI(save_config_kwargs={"overwrite": True}, run=False)
        self.config = self.zap.config.as_dict()
        self.zap.trainer.logger.log_hyperparams({'optimizer': self.config['optimizer']})
    
    def fit(self):
        self.zap.trainer.fit(self.zap.model, self.zap.datamodule)

    def test(self):
        self.zap.trainer.test(self.zap.model, self.zap.datamodule, ckpt_path="best")

    def predict(self):
        preds = self.zap.trainer.predict(self.zap.model, self.zap.datamodule, return_predictions=True, ckpt_path="last")
        return preds
    
    
        

        