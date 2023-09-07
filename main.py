from lightning.pytorch.cli import LightningCLI


def cli_main():    
    cli = LightningCLI(save_config_kwargs={"overwrite": True}, run=False)
    
    config = cli.config.as_dict()
    
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
    
    preds = cli.trainer.predict(cli.model, cli.datamodule, return_predictions=True, ckpt_path="last")
    
    cli.trainer.logger.log_hyperparams({'optimizer': config['optimizer']})

if __name__ == '__main__':
    cli_main()