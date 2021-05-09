import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .ColorizeDataset import ColorizeDataset
from ..utils import output_dir


class ColorizeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, image_size=(224, 224)):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage=None):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.image_size)
            ])
        }
        self.train = ColorizeDataset(
            output_dir / 'train', data_transforms['test'])
        self.val = ColorizeDataset(output_dir / 'val', data_transforms['val'])
        self.test = ColorizeDataset(
            output_dir / 'test', data_transforms['test'])

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)
