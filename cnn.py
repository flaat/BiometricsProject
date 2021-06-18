import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.nn.functional import relu, softmax, log_softmax
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, F1
from torchvision import models


from gallery import Gallery
from filters.gabor import build_filters


class ImageDataset(Dataset):
    def __init__(self, path):
        super(ImageDataset, self).__init__()
        g = Gallery()
        g.build_gallery(path)
        self.images, self.labels = g.get_all_original_template(mode="coded")
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            img = img.copy()
            self.images[i] = (torch.as_tensor(img).float() / 255).reshape(1, *img.shape)
            self.labels[i] = torch.as_tensor(label)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.images)


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 20, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, images):
        return self.mlp(images)


class Cnn(nn.Module):
    def __init__(self, num_classes):
        super(Cnn, self).__init__()
        self.conv_layers = nn.Sequential(
            # (1, 60, 60)
            self._conv_block(1, 64),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # (64, 30, 30)
            self._conv_block(64, 128),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # (128, 15, 15)
        )
        self.linear = nn.Linear(128 * 15 * 15, num_classes)

    def _conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(padding, padding),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, images):
        out = self.conv_layers(images)
        out = out.view(-1, 128 * 15 * 15)
        logits = self.linear(out)
        return logits


class CnnGabor(nn.Module):
    def __init__(self, num_classes):
        super(CnnGabor, self).__init__()
        gabor_filters = build_filters(3)
        # fix gabor filters
        self.conv_gabor = nn.Conv2d(
            1, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_gabor.weight = nn.Parameter(torch.tensor(gabor_filters).unsqueeze(1))

        self.conv_layers = nn.Sequential(
            # (12, 60, 60)
            self._conv_block(12, 64),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # (64, 30, 30)
            self._conv_block(64, 128),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            # (128, 15, 15)
            self._conv_block(128, 16),
            nn.Dropout(0.3),
        )
        self.linear = nn.Linear(16 * 15 * 15, num_classes)

    def _conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(padding, padding),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, images):
        with torch.no_grad():
            out = self.conv_gabor(images)
        out = relu(out)
        out = self.conv_layers(out)
        out = out.view(-1, 16 * 15 * 15)
        return self.linear(out)


class PlImageDataset(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PlImageDataset, self).__init__()
        self.hparams = hparams

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            ImageDataset(self.hparams["train_path"]),
            batch_size=self.hparams["batch_size"],
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ImageDataset(self.hparams["val_path"]),
            batch_size=self.hparams["batch_size"],
            shuffle=False,
        )


class PlModel(pl.LightningModule):
    def __init__(self, hparams):
        super(PlModel, self).__init__()
        self.save_hyperparameters(hparams)
        if self.hparams["use_resnet"]:
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.hparams["num_classes"])
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.model = model
        else:
            CnnModel = CnnGabor if hparams["use_gabor"] else Cnn
            self.model = CnnModel(self.hparams["num_classes"])

        self.criterion = nn.CrossEntropyLoss()
        self.train_f1 = F1(num_classes=self.hparams["num_classes"], average="macro")
        self.val_f1 = F1(num_classes=self.hparams["num_classes"], average="macro")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def step(self, batch: Dict[str, torch.Tensor]):
        images = batch["images"]
        labels = batch["labels"]
        output = self.model(images)
        predictions = torch.argmax(output, dim=-1)
        loss = self.criterion(output, labels)
        return {"loss": loss, "predictions": predictions}

    def training_step(self, batch, batch_idx=None):
        output = self.step(batch)
        train_loss = output["loss"]
        self.log_dict({"train_loss": train_loss}, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx=None):
        labels = batch["labels"]
        output = self.step(batch)
        predictions = output["predictions"]
        val_loss = output["loss"]
        self.log_dict({"val_loss": val_loss}, prog_bar=True)
        return {"labels": labels, "predictions": predictions}

    def validation_epoch_end(self, outputs):
        y_true_torch = torch.cat([x["labels"] for x in outputs])
        y_pred_torch = torch.cat([x["predictions"] for x in outputs])
        f1 = self.val_f1(y_true_torch, y_pred_torch)
        self.val_f1.reset()
        self.log_dict({"val_f1": f1}, prog_bar=True)


if __name__ == "__main__":
    pl.seed_everything(42)
    train_path = "data/train"
    val_path = "data/val"
    num_classes = len(os.listdir(train_path))
    hparams = dict(
        train_path=train_path,
        val_path=val_path,
        num_classes=num_classes,
        batch_size=16,
        lr=0.0001,
        weight_decay=0.0001,
        use_gabor=False,
        use_resnet=False,
    )
    model = PlModel(hparams)
    data_module = PlImageDataset(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath="cnn_checkpoints/",
        filename="{epoch}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_last=True,
        save_top_k=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_f1",
        patience=15,
        mode="max",
    )

    callbacks = [checkpoint_callback, early_stopping_callback]
    trainer = pl.Trainer(gpus=1, max_epochs=1000, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)
