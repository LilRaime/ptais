from pathlib import Path
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25, device="cpu", save_path="best_model.pth"):
    device = torch.device(device)
    model.to(device)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": 0, "best_val_acc": 0.0}
    best_val_acc, best_val_loss = -1.0, float("inf")

    print(f"\n[+] Starting training on {device} ({epochs} epochs, batch {train_loader.batch_size})")
    print("-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^11} | {'Val Acc':^11} | Status")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += len(y)

        ep_tr_loss, ep_tr_acc = tr_loss / tr_total, tr_correct / tr_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(y)

        ep_val_loss, ep_val_acc = val_loss / val_total, val_correct / val_total
        history["train_loss"].append(ep_tr_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)

        status = ""
        if (ep_val_acc > best_val_acc) or (ep_val_acc == best_val_acc and ep_val_loss < best_val_loss):
            best_val_acc, best_val_loss = ep_val_acc, ep_val_loss
            history["best_epoch"], history["best_val_acc"] = epoch, best_val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": ep_val_acc}, save_path)
            status = f"Saved (Val Acc: {ep_val_acc * 100:.1f}%)"

        print(f"{epoch:^7d} | {ep_tr_loss:^12.4f} | {ep_tr_acc * 100:^10.2f}% | {ep_val_loss:^11.4f} | {ep_val_acc * 100:^10.2f}% | {status}")

    print("-" * 65)
    print(f"[✓] Best checkpoint: {Path(save_path).resolve()} (Val Acc: {history['best_val_acc'] * 100:.2f}%, Epoch {history['best_epoch']})")
    return history
