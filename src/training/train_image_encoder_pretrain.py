import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct_dataset import OCTDataset, get_transforms, collate_fn_image_only
from src.models.image_encoder import ImageEncoder
from src.utils.seed import set_seed, SEED


# ---------- config ----------

class Config:
    data_root = "data/raw"
    train_csv = "data/splits/train.csv"
    val_csv = "data/splits/val.csv"

    num_classes = 8
    img_size = 224
    patch_size = 32
    embed_dim = 384
    depth = 6
    heads = 6
    out_dim = 256

    bs = 32
    epochs = 30
    lr = 5e-4
    wd = 1e-4
    label_smooth = 0.1
    warmup = 3
    grad_clip = 1.0
    patience = 8

    drop = 0.0
    path_drop = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0
    pin_mem = True

    save_dir = "experiments/image_encoder_pretrain"
    resume = None
    # resume = "experiments/image_encoder_pretrain/ckpts/last.pth"


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ---------- model ----------

class ImageClassifier(nn.Module):

    def __init__(self, n_classes=8):
        super().__init__()
        self.encoder = ImageEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            out_dim=cfg.out_dim,
            drop=cfg.drop,
            path_drop=cfg.path_drop,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.out_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, images):
        emb = self.encoder(images)
        return self.head(emb), emb


# ---------- data ----------

def make_loaders():
    train_ds = OCTDataset(
        csv_path=cfg.train_csv,
        data_root=cfg.data_root,
        prompts_path=None,
        transform=get_transforms("train", cfg.img_size),
        tokenizer=None,
        mode="train",
    )
    val_ds = OCTDataset(
        csv_path=cfg.val_csv,
        data_root=cfg.data_root,
        prompts_path=None,
        transform=get_transforms("eval", cfg.img_size),
        tokenizer=None,
        mode="eval",
    )

    shared = dict(
        num_workers=cfg.workers,
        pin_memory=cfg.pin_mem,
        collate_fn=collate_fn_image_only,
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, **shared)
    val_dl = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, **shared)
    return train_dl, val_dl


# ---------- train / val loops ----------

def run_train(model, loader, loss_fn, opt, scaler, ep):
    model.train()
    running_loss = 0.0
    hits, seen = 0, 0

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [train]")
    for imgs, labels in pbar:
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device)

        opt.zero_grad()

        with autocast(cfg.device, enabled=cfg.amp):
            logits, _ = model(imgs)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        hits += logits.argmax(1).eq(labels).sum().item()
        seen += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * hits / seen:.1f}%")

    avg_loss = running_loss / len(loader)
    acc = 100.0 * hits / seen
    return avg_loss, acc


@torch.no_grad()
def run_val(model, loader, loss_fn, ep):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [val]"):
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            logits, _ = model(imgs)
            loss = loss_fn(logits, labels)

        running_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    truth = np.array(all_labels)

    avg_loss = running_loss / len(loader)
    acc = 100.0 * (preds == truth).mean()
    f1 = f1_score(truth, preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


# ---------- plots / reports ----------

CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]


def save_plots(hist, preds, labels):
    ep = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ep, hist["train_loss"], label="Train", marker="o", ms=3)
    axes[0].plot(ep, hist["val_loss"], label="Val", marker="o", ms=3)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist["train_acc"], label="Train", marker="o", ms=3)
    axes[1].plot(ep, hist["val_acc"], label="Val", marker="o", ms=3)
    axes[1].set(xlabel="Epoch", ylabel="Acc (%)", title="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, hist["lr"], marker="o", ms=3, color="tab:red")
    axes[2].set(xlabel="Epoch", ylabel="LR", title="LR Schedule")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/training_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    with open(f"{cfg.save_dir}/classification_report.txt", "w") as f:
        f.write(f"LR:{cfg.lr} LS:{cfg.label_smooth}\n{'=' * 70}\n\n{report}")
    print(report)


# ---------- main ----------

def main():
    print(f"{'=' * 70}")
    print("  STAGE 1: IMAGE ENCODER PRE-TRAINING")
    print(f"{'=' * 70}")
    print(f"  {cfg.epochs}ep | LR={cfg.lr} | LS={cfg.label_smooth} | DropPath={cfg.path_drop}")

    set_seed()

    train_dl, val_dl = make_loaders()
    print(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")

    model = ImageClassifier().to(cfg.device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
    )

    warmup_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(
        opt, T_max=cfg.epochs - cfg.warmup, eta_min=1e-6,
    )
    sched = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[cfg.warmup])

    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    start_ep = 0
    best_f1 = 0.0
    wait = 0
    hist = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": [], "lr": [],
    }

    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        wait = ckpt["wait"]
        hist = ckpt["hist"]
        print(f"  Resumed from epoch {start_ep}, best F1: {best_f1:.4f}")

    for ep in range(start_ep, cfg.epochs):
        t_loss, t_acc = run_train(model, train_dl, loss_fn, opt, scaler, ep)
        v_loss, v_acc, v_f1, preds, labels = run_val(model, val_dl, loss_fn, ep)
        sched.step()

        cur_lr = opt.param_groups[0]["lr"]

        hist["train_loss"].append(t_loss)
        hist["train_acc"].append(t_acc)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)
        hist["val_f1"].append(v_f1)
        hist["lr"].append(cur_lr)

        print(
            f"\nEp {ep + 1}/{cfg.epochs}: "
            f"Train {t_acc:.1f}% | Val {v_acc:.1f}% F1={v_f1:.4f} LR={cur_lr:.2e}"
        )

        if v_f1 > best_f1:
            best_f1 = v_f1
            wait = 0
            print(f"  Best F1: {best_f1:.4f}")
            torch.save(
                model.encoder.state_dict(),
                f"{cfg.save_dir}/ckpts/best_encoder.pth",
            )
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "wait": wait,
            "hist": hist,
        }, f"{cfg.save_dir}/ckpts/last.pth")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {ep + 1}")
            break

    torch.save(
        model.encoder.state_dict(),
        f"{cfg.save_dir}/ckpts/final_encoder.pth",
    )
    pd.DataFrame(hist).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(hist, preds, labels)

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best F1: {best_f1:.4f} | Acc: {max(hist['val_acc']):.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()