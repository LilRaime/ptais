import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.dataset import create_dataloaders
from src.evaluate import evaluate_model
from src.model import SimpleCNN
from src.train import train_model
from src.visualization import plot_confusion_matrix, plot_predictions, plot_training_curves

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CNN (Conv->ReLU->Pool x2 -> AdaptiveAvgPool -> Linear) for Cat/Dog/Raven")
    default_ds = Path("../lab1/dataset") if Path("../lab1/dataset").exists() else Path("lab1/dataset")
    parser.add_argument("--dataset-dir", type=Path, default=default_ds)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=Path, default=Path("best_model.pth"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-preload", action="store_true", help="Disable RAM caching of dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit("[bold cyan]CV CNN Pipeline (Lab 2)[/bold cyan]\n[dim]Cat / Dog / Raven Classification [/dim]", border_style="bright_blue"))

    print(f"\n[+] Loading dataset from: {args.dataset_dir.resolve()} (RAM preload: {not args.no_preload})")
    train_loader, val_loader, test_loader, class_names, (ds_tr, ds_val, ds_test) = create_dataloaders(
        args.dataset_dir, args.batch_size, args.seed, preload=not args.no_preload
    )
    total = len(ds_tr) + len(ds_val) + len(ds_test)

    t_split = Table(title="Dataset Partitioning", border_style="bright_blue")
    t_split.add_column("Subset", style="bold cyan"); t_split.add_column("Count", justify="right", style="green"); t_split.add_column("Share", justify="right", style="yellow")
    for name, ds in [("Train", ds_tr), ("Validation", ds_val), ("Test", ds_test)]:
        t_split.add_row(name, str(len(ds)), f"{len(ds)/total:.1%}")
    console.print(t_split)

    model = SimpleCNN(num_classes=len(class_names))
    print(f"[Total trainable parameters: {model.count_parameters():,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.device, args.save_model)
    plot_training_curves(history, args.plots_dir / "loss_accuracy_curves.png", args.show_plots)

    checkpoint = torch.load(args.save_model, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    acc, prec, rec, cm, rep, correct, incorrect = evaluate_model(model, test_loader, class_names, args.device, val_loader)

    t_metrics = Table(title="Test Set Evaluation Metrics", border_style="bright_green")
    t_metrics.add_column("Class", style="bold cyan"); t_metrics.add_column("Precision", justify="right"); t_metrics.add_column("Recall", justify="right"); t_metrics.add_column("F1", justify="right"); t_metrics.add_column("Support", justify="right")
    for c in class_names:
        info = rep.get(c, {})
        t_metrics.add_row(c, f"{info.get('precision', 0)*100:.1f}%", f"{info.get('recall', 0)*100:.1f}%", f"{info.get('f1-score', 0)*100:.1f}%", str(int(info.get('support', 0))))
    t_metrics.add_section()
    t_metrics.add_row("Accuracy", "-", "-", f"[bold]{acc * 100:.2f}%[/bold]", str(len(ds_test)))
    t_metrics.add_row("Macro Avg", f"{prec * 100:.1f}%", f"{rec * 100:.1f}%", f"{rep.get('macro avg', {}).get('f1-score', 0)*100:.1f}%", "-")
    console.print(t_metrics)

    plot_confusion_matrix(cm, class_names, args.plots_dir / "confusion_matrix.png", args.show_plots)
    plot_predictions(correct, incorrect, args.plots_dir / "predictions_analysis.png", args.show_plots)

    t_err = Table(title="Top Misclassifications & Explanations", border_style="red")
    t_err.add_column("True", style="bold red"); t_err.add_column("Pred", style="bold yellow"); t_err.add_column("Subset", style="cyan"); t_err.add_column("File"); t_err.add_column("Explanation")
    for s in incorrect:
        t_err.add_row(s.true_name, s.pred_name, s.subset, Path(s.path).name, s.explanation)
    console.print(t_err)

    print("\nPipeline complete! Checkpoints and plots saved.")

if __name__ == "__main__":
    main()
