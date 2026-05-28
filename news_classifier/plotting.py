from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plots(trainer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logged = trainer.logged_metrics

    if not hasattr(trainer, "callback_metrics"):
        print("No metrics to plot")
        return

    metric_history = {}
    for callback in trainer.callbacks:
        if hasattr(callback, "state_dict"):
            state = callback.state_dict()
            if "best_model_score" in state:
                metric_history["best_val_f1"] = state["best_model_score"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    val = logged.get("train_loss", "N/A")
    ax.text(0.5, 0.5, f"Final: {val}", transform=ax.transAxes, ha="center")

    ax = axes[1]
    ax.set_title("Validation F1 (Macro)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.text(0.5, 0.5, f"Final: {logged.get('val_f1', 'N/A')}", transform=ax.transAxes, ha="center")

    ax = axes[2]
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.text(0.5, 0.5, f"Final: {logged.get('val_acc', 'N/A')}", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=100)
    plt.close()
    print(f"Plots saved to {output_dir}")
