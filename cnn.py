import os
import torch
import pytorch_lightning as pl
from typing import Dict

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from gallery import Gallery
from filters.gabor import build_filters


class ImageDataset(Dataset):
    def __init__(self, path):
        super(ImageDataset, self).__init__()
        g = Gallery()
        g.build_gallery(path)
        self.images, self.labels = g.get_all_original_template()
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            img = img.copy()
            self.images[i] = (
                torch.as_tensor(img, dtype=torch.float32).unsqueeze(0) / 255
            )
            self.labels[i] = torch.as_tensor(label)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.images)


class Cnn(nn.Module):
    # (1, 60, 60)
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
        return self.linear(out)


class CnnGabor(nn.Module):
    def __init__(self):
        super(CnnGabor, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


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
        self.model = Cnn(self.hparams["num_classes"])
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(
            num_classes=self.hparams["num_classes"], average="macro"
        )
        self.val_accuracy = Accuracy(
            num_classes=self.hparams["num_classes"], average="macro"
        )

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
        labels = batch["labels"]
        output = self.step(batch)
        predictions = output["predictions"]
        train_loss = output["loss"]
        train_accuracy = self.train_accuracy(predictions, labels)
        self.log_dict(
            {"train_loss": train_loss, "train_accuracy": train_accuracy},
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx=None):
        labels = batch["labels"]
        output = self.step(batch)
        predictions = output["predictions"]
        val_loss = output["loss"]
        val_accuracy = self.val_accuracy(predictions, labels)
        self.log_dict(
            {"val_loss": val_loss, "val_accuracy": val_accuracy}, prog_bar=True
        )


if __name__ == "__main__":
    train_path = "celeba/train_"
    val_path = "celeba/val_"
    num_classes = len(os.listdir(train_path))
    hparams = dict(
        train_path=train_path,
        val_path=val_path,
        num_classes=num_classes,
        batch_size=32,
        lr=0.002,
        weight_decay=0.0,
    )
    model = PlModel(hparams)
    data_module = PlImageDataset(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath="cnn_checkpoints/",
        filename="{epoch}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        mode="max",
        save_last=True,
        save_top_k=3,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
    )

    callbacks = [checkpoint_callback, early_stopping_callback]
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)
