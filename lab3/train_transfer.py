import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table

from src.dataset import create_dataloaders
from src.models import get_model, count_parameters

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_ds = Path("../lab1/dataset") if Path("../lab1/dataset").exists() else Path("lab1/dataset")
    parser.add_argument("--dataset-dir", type=Path, default=default_ds)
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("weights/mobilenet_v3_finetuned.pth"))
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def main():
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names, (ds_tr, ds_val, ds_test) = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        seed=args.seed,
        preload=not args.no_preload
    )

    model = get_model(
        name=args.model_name,
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=not args.unfreeze_backbone
    ).to(args.device)

    trainable_params = count_parameters(model)
    console.print(f"[bold cyan]Model:[/bold cyan] {args.model_name} | [bold green]Trainable Parameters:[/bold green] {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    trainable_layers = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_layers, lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    table = Table(title=f"Transfer Learning Training ({args.model_name})")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Train Acc", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Val Acc", justify="right")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        table.add_row(
            str(epoch),
            f"{tr_loss:.4f}",
            f"{tr_acc * 100:.2f}%",
            f"{val_loss:.4f}",
            f"{val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "img_size": args.img_size
            }, args.save_path)

    console.print(table)
    checkpoint = torch.load(args.save_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    console.print(f"[bold green]Best Validation Accuracy:[/bold green] {best_val_acc * 100:.2f}%")
    console.print(f"[bold yellow]Final Test Accuracy:[/bold yellow] {test_acc * 100:.2f}%")
    console.print(f"Saved checkpoint to: {args.save_path}")

if __name__ == "__main__":
    main()
