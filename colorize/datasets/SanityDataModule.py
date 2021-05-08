import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .ColorizeDataset import ColorizeDataset
from ..utils import output_dir


class SanityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, image_size=(224, 224)):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage=None):
        data_transforms = transforms.Compose([
            transforms.Resize(self.image_size)
        ])
        self.train = ColorizeDataset(output_dir / 'sanity', data_transforms)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.train, self.batch_size)
