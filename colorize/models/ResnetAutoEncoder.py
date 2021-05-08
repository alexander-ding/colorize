import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import pytorch_lightning as pl


class ResnetAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        # replace first layer to take in single-channel inputs
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.features = nn.Sequential(
            *list(resnet.children())[0:5])  # 256 x 56 x 56

        self.upsample = nn.Sequential(
            nn.Conv2d(256, 128, 3),  # 128 x 54 x 54
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 128 x 108 x 108
            nn.Conv2d(128, 64, 3),  # 64 x 106 x 106
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample((226, 226)),  # 64 x 226 x 226
            nn.Conv2d(64, 2, 3),  # 2 x 224 x 224
            nn.Sigmoid(),
        )

    def forward(self, x):
        assert(x.dim() in [3, 4])
        assert(x.shape[-1] == 224 and x.shape[-2] == 224)
        if x.dim() == 3:
            x = x.reshape(x.shape[0], 1, 224, 224)
        else:  # x.dim() == 4
            assert(x.shape[1] == 1)

        features = self.features(x)
        return self.upsample(features)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)
        return optimizer  # [optimizer], [lr_scheduler]
