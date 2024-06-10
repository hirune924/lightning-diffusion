import lightning as L
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import numpy as np
from typing import Any

class VisualizationCallback(L.Callback):
    def __init__(self,
                 prompt: list[str],
                 interval: int = 1,
                 height: int | None = None,
                 width: int | None = None,
                 by_epoch: bool = True,
                 **kwargs) -> None:
        self.prompt = prompt
        self.kwargs = kwargs
        self.interval = interval
        self.by_epoch = by_epoch
        self.height = height
        self.width = width

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.generate_and_log(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.by_epoch:
            return
        if (trainer.global_step + 1) % self.interval == 0:
            self.generate_and_log(trainer, pl_module)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if self.by_epoch and (trainer.current_epoch + 1) % self.interval == 0:
            self.generate_and_log(trainer, pl_module)

    def log_images(self, logger: Logger, images: np.ndarray, steps: int = 0):
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_images("generated_images",
                                         images, 
                                         steps, 
                                         dataformats='NHWC')
        elif isinstance(logger, WandbLogger):
            logger.log_image(key="generated_images",
                                        images=[i for i in images],
                                        step=steps)
        elif isinstance(logger, CSVLogger):
            pass
        else:
            raise NotImplementedError()

    def generate_and_log(self, trainer: L.Trainer, pl_module: L.LightningModule):
        images = pl_module(
            prompt = self.prompt,
            height=self.height,
            width=self.width,
            **self.kwargs)
        loggers = trainer.loggers
        if isinstance(loggers, list):
            for l in loggers:
                self.log_images(l, np.stack(images), trainer.global_step)
        else:
            self.log_images(loggers, np.stack(images), trainer.global_step)
