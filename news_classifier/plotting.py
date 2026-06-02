from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class MetricsTracker(pl.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_f1 = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_loss.append(float(loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key, store in [
            ("val_loss", self.val_loss),
            ("val_f1", self.val_f1),
            ("val_acc", self.val_acc),
        ]:
            val = metrics.get(key)
            if val is not None:
                store.append(float(val))


def _plot_curve(values, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def save_training_plots(tracker, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if not tracker.train_loss:
        print("nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(tracker.train_loss) + 1)
    ax.plot(epochs, tracker.train_loss, label="train")
    if tracker.val_loss:
        ax.plot(range(1, len(tracker.val_loss) + 1), tracker.val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=100)
    plt.close()

    if tracker.val_f1:
        _plot_curve(tracker.val_f1, "Macro F1", "Validation F1", output_dir / "val_f1.png")

    if tracker.val_acc:
        _plot_curve(
            tracker.val_acc, "Accuracy", "Validation Accuracy", output_dir / "val_accuracy.png"
        )

    print(f"saved plots to {output_dir}")
