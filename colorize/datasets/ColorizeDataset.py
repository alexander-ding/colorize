from pathlib import Path
from PIL import Image
import skimage.color as color
import torch
import numpy as np
from torch.utils.data import Dataset


class ColorizeDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_names = list(Path(image_path).glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        im = Image.open(image_name)
        if self.transform:
            im = self.transform(im)
        im = np.asarray(im) / 255
        im = color.rgb2lab(im)
        im = torch.as_tensor(im, dtype=torch.float32).permute((2, 0, 1))
        l_channel = torch.unsqueeze(im[0], 0) / 100
        ab_channels = (im[1:] + 127) / 255
        return l_channel, ab_channels

    @staticmethod
    def create_lab(x, y):
        return np.concatenate([x * 100, y * 255 - 127], axis=0).transpose((1, 2, 0))

    @staticmethod
    def lab2rgb(lab):
        return color.lab2rgb(lab)
