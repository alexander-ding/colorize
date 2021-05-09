import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch
import pytorch_lightning as pl


class ResnetFCN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.fcn = torch.hub.load(
            'pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
        # replace first layer to take in single-channel inputs
        self.fcn.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.fcn.classifier = nn.Sequential(
            *list(self.fcn.classifier.children())[:-1],
            nn.Conv2d(512, 2, 1, 1),
            nn.Sigmoid(),
        )
        self.fcn.aux_classifier = nn.Sequential(
            *list(self.fcn.aux_classifier.children())[:-1],
            nn.Conv2d(256, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        assert(x.dim() in [3, 4])
        assert(x.shape[-1] == 224 and x.shape[-2] == 224)
        if x.dim() == 3:
            x = x.reshape(x.shape[0], 1, 224, 224)
        else:  # x.dim() == 4
            assert(x.shape[1] == 1)

        return self.fcn(x)['out']

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.l1_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.l1_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.l1_loss(y_pred, y_true)
        # logging to TensorBoard
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer  # [optimizer], [lr_scheduler]
