from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from colorize.models import *
from colorize.datasets import *

model = UNetGAN()  # ResnetFCN() # ResnetRGBAutoEncoder()
data = ColorizeDataModule(batch_size=8, image_size=(256, 256))

logger = pl.loggers.tensorboard.TensorBoardLogger(
    'lightning_logs', name='GAN', version=None
)

trainer = pl.trainer.Trainer(callbacks=[],  # [EarlyStopping(monitor='val_loss')],
                             max_epochs=100,
                             gpus=0,
                             logger=logger)

trainer.fit(model, datamodule=data)
