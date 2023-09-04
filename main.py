import mlflow
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from torchvision.transforms import Compose, Resize, ToTensor

from config import TrainConfig
from data_modules import CustomDataModule
from models import MyResnetModel
from utils import get_label_map

# RANDOM_STATE = 42  # TODO: read from env?
# seed_everything(RANDOM_STATE, workers=True)  # NOTE: this freezes experiment names

# mlflow.pytorch.autolog()  # NOTE: seems to record/register model but doesn't autolog metrics; also logs to the default experiment!
# mlf_logger = MLFlowLogger(experiment_name="petars_sick_experiment", tags={'tagA': 1, 'tagB': 99}, log_model=True)  # BUG: not registering model atm

# transforms = Compose([ToTensor(), Resize((32, 32))])
# dm = CustomDataModule(data_dir='data', transforms=transforms, split=[0.7, 0.2, 0.1], batch_size=64)

# TODO: add weights to loss func
# weights = dm.compute_class_weights()
# weights = [w for w in weights.values()]
# weights = torch.FloatTensor(weights)
# # weights = weights.to(DEVICE)


# TODO: replace
# cfg = TrainConfig()
# label_map, label_map_reversed = get_label_map(cfg.label_map_path)

# loss_fn = torch.nn.CrossEntropyLoss()
# model = MyResnetModel(loss_fn=loss_fn, lr=1e-4, momentum=0, label_map=label_map, bias=True)
# trainer = Trainer(max_epochs=3, profiler=None, logger=mlf_logger)

# if __name__ == '__main__':
#     trainer.fit(model, datamodule=dm)
#     trainer.validate(model, datamodule=dm)
#     trainer.test(model, datamodule=dm)


def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
    cli_main()
