from lightning.pytorch.callbacks import Callback


class RegisterModelCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        for d in dir(self):
            print(d)

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        self.log('aa', 11)
