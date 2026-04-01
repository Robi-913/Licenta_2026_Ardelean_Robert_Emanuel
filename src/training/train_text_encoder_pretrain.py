import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.text_encoder import TextEncoder
from src.utils.seed import set_seed, SEED


# ---------- config ----------

class Config:
    prompts_file = "data/old/prompts_expanded_structured.json"

    num_classes = 4
    vocab_size = 30522
    max_len = 77
    embed_dim = 256
    depth = 4
    heads = 4
    out_dim = 256

    bs = 64
    epochs = 20
    lr = 1e-4
    wd = 1e-3
    label_smooth = 0.15
    warmup = 2
    grad_clip = 1.0
    patience = 6
    drop = 0.1

    target_train = 12000
    target_val = 1500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0
    pin_mem = False

    save_dir = "experiments/text_encoder_pretrain"


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ---------- dataset ----------

class PromptDataset(Dataset):

    def __init__(self, prompts_file, tokenizer, class_names, split="train"):
        self.tokenizer = tokenizer
        self.class_names = class_names

        with open(prompts_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        for i, cls_name in enumerate(self.class_names):
            entry = raw[cls_name]
            prompts = entry["all"] if isinstance(entry, dict) else entry
            for p in prompts:
                self.samples.append((p, i))

        print(f"  Unique prompts: {len(self.samples)}")

        np.random.seed(SEED)
        order = np.random.permutation(len(self.samples))
        cutoff = int(0.9 * len(self.samples))

        if split == "train":
            self.samples = [self.samples[j] for j in order[:cutoff]]
            reps = max(1, cfg.target_train // len(self.samples))
            self.samples = self.samples * reps
            print(f"  Train: {len(self.samples)} samples ({reps}x)")
        else:
            self.samples = [self.samples[j] for j in order[cutoff:]]
            reps = max(1, cfg.target_val // len(self.samples))
            self.samples = self.samples * reps
            print(f"  Val:   {len(self.samples)} samples ({reps}x)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_len,
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        return ids, mask, label


# ---------- model ----------

class TextClassifier(nn.Module):

    def __init__(self, n_classes=4):
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size=cfg.vocab_size,
            max_len=cfg.max_len,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            out_dim=cfg.out_dim,
            pool="mean",
            drop=cfg.drop,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.out_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask)
        return self.head(emb), emb


# ---------- train / val ----------

def run_train(model, loader, loss_fn, opt, scaler, ep):
    model.train()
    running_loss = 0.0
    hits, seen = 0, 0

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [train]")
    for ids, mask, labels in pbar:
        ids = ids.to(cfg.device)
        mask = mask.to(cfg.device)
        labels = labels.to(cfg.device)

        opt.zero_grad()

        with autocast(cfg.device, enabled=cfg.amp):
            logits, _ = model(ids, mask)
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

    return running_loss / len(loader), 100.0 * hits / seen


@torch.no_grad()
def run_val(model, loader, loss_fn, ep):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for ids, mask, labels in tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [val]"):
        ids = ids.to(cfg.device)
        mask = mask.to(cfg.device)
        labels = labels.to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            logits, _ = model(ids, mask)
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


# ---------- plots ----------

def save_plots(hist, preds, labels, class_names):
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    with open(f"{cfg.save_dir}/classification_report.txt", "w") as f:
        f.write(f"LR:{cfg.lr} LS:{cfg.label_smooth} Drop:{cfg.drop}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(report)
    print(report)


# ---------- main ----------

def main():
    print(f"{'=' * 70}")
    print("  STAGE 2: TEXT ENCODER PRE-TRAINING")
    print(f"{'=' * 70}")
    print(f"  {cfg.epochs}ep | LR={cfg.lr} | LS={cfg.label_smooth} | Drop={cfg.drop}")

    set_seed()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    with open(cfg.prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    class_names = sorted(prompts_data.keys())
    cfg.num_classes = len(class_names)

    train_ds = PromptDataset(cfg.prompts_file, tokenizer, class_names, "train")
    val_ds = PromptDataset(cfg.prompts_file, tokenizer, class_names, "val")

    loader_kw = dict(num_workers=cfg.workers, pin_memory=cfg.pin_mem)
    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, **loader_kw)

    model = TextClassifier(n_classes=cfg.num_classes).to(cfg.device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    warmup_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=1e-6)
    sched = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[cfg.warmup])

    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    best_f1 = 0.0
    wait = 0
    hist = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": [], "lr": [],
    }

    for ep in range(cfg.epochs):
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
    save_plots(hist, preds, labels, class_names)

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best F1: {best_f1:.4f} | Acc: {max(hist['val_acc']):.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()