import subprocess
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from news_classifier.data import get_dataloaders, load_glove
from news_classifier.plotting import MetricsTracker, save_training_plots
from news_classifier.trainer import TextCNNModule


def compute_class_weights(train_loader):
    label_counts = {}
    for _, labels in train_loader:
        for label in labels.tolist():
            label_counts[label] = label_counts.get(label, 0) + 1

    total = sum(label_counts.values())
    num_classes = len(label_counts)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    return weights


def _get_git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def train(cfg):
    pl.seed_everything(cfg.training.seed)

    train_loader, val_loader, test_loader, vocab, cat_to_idx = get_dataloaders(cfg)
    class_weights = compute_class_weights(train_loader)

    pretrained = None
    glove_path = Path(cfg.model.glove_path)
    if glove_path.exists():
        pretrained = load_glove(glove_path, vocab, cfg.model.embed_dim)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
    )

    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlf_logger.experiment.log_param(mlf_logger.run_id, "git_commit", _get_git_commit())

    model = TextCNNModule(
        cfg=cfg,
        vocab_size=len(vocab),
        num_classes=len(cat_to_idx),
        class_weights=class_weights,
        pretrained_embeddings=pretrained,
    )

    tracker = MetricsTracker()
    callbacks = [
        EarlyStopping(monitor="val_f1", mode="max", patience=cfg.training.patience),
        ModelCheckpoint(
            dirpath="models",
            filename="best-{epoch}-{val_f1:.3f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        ),
        tracker,
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    save_training_plots(tracker, "plots")
