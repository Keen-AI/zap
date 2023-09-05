import mlflow
from lightning.pytorch.cli import LightningCLI

mlflow.set_experiment("my_sick_experiment")  # TODO: read from env
mlflow.pytorch.autolog(registered_model_name='my_sick_model')  # TODO: read from env


def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
    cli_main()
