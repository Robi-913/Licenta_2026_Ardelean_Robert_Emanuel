import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.cnn_resnet18 import ResNet18OCT
from src.utils.seed import set_seed


# CONFIG

class Config:
    splits_dir = "data/oct5k/splits_v3"
    img_size = 224
    n_classes = 4  # AMD / DME / DRUSEN / NORMAL
    batch_size = 32
    epochs = 30
    lr = 1e-3
    workers = 4
    patience = 8  # early stopping
    output_dir = "experiments/cnn_v2"
    ckpt_final = "checkpoints/resnet18_v2_final.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(f"{cfg.output_dir}/ckpts", exist_ok=True)

# Directoarele unde se gasesc imaginile OCT5k — acelasi sistem ca OCT5kDataset
_IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


# DATASET

def _locate(rel_path: str) -> str | None:
    """Cauta imaginea in directoarele cunoscute, cu fallback pe extensii."""
    norm = rel_path.replace("\\", "/")
    for base in _IMG_DIRS:
        full = Path(base) / norm
        if full.exists():
            return str(full)
        for ext in [".png", ".jpeg", ".jpg"]:
            alt = full.with_suffix(ext)
            if alt.exists():
                return str(alt)
    return None


def _get_transforms(mode: str) -> transforms.Compose:
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class OCT5kCNNDataset(Dataset):
    """
    Dataset simplu pentru CNN — incarca imagine + label de clasa.
    Foloseste acelasi format CSV ca OCT5kDataset (coloane: image_path, disease).
    Nu are nevoie de prompturi sau severitate.
    """

    def __init__(self, csv_path: str, mode: str = "train"):
        df = pd.read_csv(csv_path)

        # Construim labelmap din clasele unice sortate — consistent intre splits
        self.classes = sorted(df["disease"].unique())
        self.lbl_map = {name: i for i, name in enumerate(self.classes)}

        # Filtram randurile pentru care gasim imaginea pe disk
        self.samples = []
        n_missing = 0
        for _, row in df.iterrows():
            path = _locate(row["image_path"])
            if path is None:
                n_missing += 1
                continue
            self.samples.append({
                "path": path,
                "label": self.lbl_map[row["disease"]],
            })

        self.transform = _get_transforms(mode)
        print(
            f"  OCT5kCNN [{mode}]: {len(self.samples)} imagini, {n_missing} lipsa, {len(self.classes)} clase: {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # reduce speckle OCT
        return {"image": self.transform(img), "label": s["label"]}


def _collate(batch: list) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


# TRAINING & VALIDATION

def run_train_epoch(
        model: ResNet18OCT,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        epoch: int,
) -> tuple[float, float]:
    model.train()
    tot_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc=f"  Epoch {epoch}/{cfg.epochs} [Train]")
    for batch in pbar:
        images = batch["image"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = tot_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def run_val(
        model: ResNet18OCT,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
) -> tuple[float, float, float, list, list]:
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Val", leave=False):
        images = batch["image"].to(cfg.device)
        labels = batch["label"].to(cfg.device)
        outputs = model(images)
        tot_loss += criterion(outputs, labels).item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = tot_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, all_preds, all_labels


# PLOTS

def save_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="Train", marker="o")
    axes[0].plot(ep, history["val_loss"], label="Val", marker="s")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend();
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["val_acc"], label="Accuracy", marker="o", color="green")
    axes[1].plot(ep, history["val_f1"], label="F1 Macro", marker="s", color="orange")
    axes[1].set(title="Validation Metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend();
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/loss_curves.png", dpi=150)
    plt.close()
    print(f"  Curves: {cfg.output_dir}/loss_curves.png")


def save_confusion_matrix(y_true: list, y_pred: list, classes: list) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted");
    plt.ylabel("True");
    plt.title("Confusion Matrix — CNN v2 (splits_v3)")
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  CM: {cfg.output_dir}/confusion_matrix.png")


# MAIN

def main():
    set_seed(42)

    print("  CNN BASELINE v2 — ResNet18 pe splits_v3")
    print(f"  Device: {cfg.device} | Epochs: {cfg.epochs} | BS: {cfg.batch_size} | LR: {cfg.lr}")

    train_ds = OCT5kCNNDataset(f"{cfg.splits_dir}/train.csv", mode="train")
    val_ds = OCT5kCNNDataset(f"{cfg.splits_dir}/val.csv", mode="eval")
    test_ds = OCT5kCNNDataset(f"{cfg.splits_dir}/test.csv", mode="eval")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True, collate_fn=_collate)

    model = ResNet18OCT(num_classes=cfg.n_classes, use_pretrained=False).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history = dict(train_loss=[], val_loss=[], val_acc=[], val_f1=[])
    best_f1 = 0.0
    wait = 0
    best_preds = []
    best_labels = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_f1, val_preds, val_labels = run_val(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\n  Epoch {epoch}: T_loss={train_loss:.4f} T_acc={train_acc:.3f} | "
              f"V_loss={val_loss:.4f} V_acc={val_acc:.3f} V_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            best_preds = val_preds
            best_labels = val_labels
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1, "val_acc": val_acc,
                "classes": train_ds.classes,
            }, f"{cfg.output_dir}/ckpts/best.pth")
            print(f"   Best: {best_f1:.4f}")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        if wait >= cfg.patience:
            print(f"  Early stopping la epoch {epoch}")
            break

    # Evaluare pe test set cu best checkpoint
    best_ckpt = torch.load(f"{cfg.output_dir}/ckpts/best.pth", map_location=cfg.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    _, test_acc, test_f1, test_preds, test_labels = run_val(model, test_loader, criterion)
    print(f"\n  TEST SET: Acc={test_acc:.3f} | F1={test_f1:.4f}")

    # Salvari finale
    pd.DataFrame(history).to_csv(f"{cfg.output_dir}/metrics.csv", index=False)
    save_training_curves(history)
    save_confusion_matrix(test_labels, test_preds, train_ds.classes)

    os.makedirs("checkpoints", exist_ok=True)
    shutil.copy(f"{cfg.output_dir}/ckpts/best.pth", cfg.ckpt_final)

    print(f"\n  DONE! Best Val F1={best_f1:.4f} | Test F1={test_f1:.4f}")
    print(f"  Checkpoint: {cfg.ckpt_final}")
    print(f"  Metrici: {cfg.output_dir}/metrics.csv")


if __name__ == "__main__":
    main()
