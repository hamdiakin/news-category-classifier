import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from news_classifier.model import TextCNN


class TextCNNModule(pl.LightningModule):
    def __init__(
        self, cfg, vocab_size, num_classes, class_weights=None, pretrained_embeddings=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_embeddings", "class_weights"])
        self.cfg = cfg

        self.model = TextCNN(
            vocab_size=vocab_size,
            embed_dim=cfg.model.embed_dim,
            num_classes=num_classes,
            num_filters=cfg.model.num_filters,
            kernel_sizes=tuple(cfg.model.kernel_sizes),
            dropout=cfg.model.dropout,
            pretrained_embeddings=pretrained_embeddings,
        )

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_weighted_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        self.log("val_loss", loss, prog_bar=True)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_weighted_f1(preds, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_weighted_f1", self.val_weighted_f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }
