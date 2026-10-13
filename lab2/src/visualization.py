from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

def plot_training_curves(history, save_path=None, show=False):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

    ax_l.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2, ms=3)
    ax_l.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e", lw=2, ms=3)
    ax_l.set(title="Training & Validation Loss", xlabel="Epoch", ylabel="Loss")
    ax_l.grid(True, ls="--", alpha=0.5)
    ax_l.legend()

    ax_a.plot(epochs, [v * 100 for v in history["train_acc"]], "o-", label="Train Acc", color="#2ca02c", lw=2, ms=3)
    ax_a.plot(epochs, [v * 100 for v in history["val_acc"]], "s-", label="Val Acc", color="#d62728", lw=2, ms=3)
    best_ep, best_val = history["best_epoch"], history["best_val_acc"] * 100
    ax_a.scatter([best_ep], [best_val], color="#d62728", s=80, zorder=5, label=f"Best: {best_val:.1f}% (Ep {best_ep})")
    ax_a.set(title="Training & Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(-2, 105))
    ax_a.grid(True, ls="--", alpha=0.5)
    ax_a.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved curves: {Path(save_path).resolve()}")
    if show: plt.show()
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_perc = np.divide(cm.astype(float), cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0) * 100
    annot = np.array([[f"{cnt}\n({pct:.1f}%)" for cnt, pct in zip(row_c, row_p)] for row_c, row_p in zip(cm, cm_perc)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title="Test Confusion Matrix", xlabel="Predicted Class", ylabel="True Class")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved confusion matrix: {Path(save_path).resolve()}")
    if show: plt.show()
    plt.close(fig)

def plot_predictions(correct, incorrect, save_path=None, show=False):
    cols = max(len(correct), len(incorrect), 1)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.2, 7.2), dpi=150)
    if cols == 1: axes = np.array([[axes[0]], [axes[1]]])

    for col in range(cols):
        ax = axes[0, col]
        if col < len(correct):
            s = correct[col]
            ax.imshow(Image.open(s.path).convert("RGB"))
            ax.set_title(f"{s.true_name.upper()}\n{s.confidence * 100:.1f}% ({s.subset})", color="#1b5e20", fontsize=9, fontweight="bold")
            for sp in ax.spines.values(): sp.set(color="#2e7d32", linewidth=2.5)
        else: ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    for col in range(cols):
        ax = axes[1, col]
        if col < len(incorrect):
            s = incorrect[col]
            ax.imshow(Image.open(s.path).convert("RGB"))
            ax.set_title(f"[✗] {s.true_name} -> {s.pred_name}\n{s.confidence * 100:.1f}% ({s.subset})", color="#b71c1c", fontsize=8.5, fontweight="bold")
            ax.set_xlabel(s.explanation, fontsize=7, wrap=True)
            for sp in ax.spines.values(): sp.set(color="#c62828", linewidth=2.5)
        else: ax.axis("off")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Predictions Analysis (Top: Correct | Bottom: Incorrect)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved predictions plot: {Path(save_path).resolve()}")
    if show: plt.show()
    plt.close(fig)
