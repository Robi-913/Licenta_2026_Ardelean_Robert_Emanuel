import os
import sys
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct_dataset import OCTDataset, get_transforms
from src.models.siglip_model import SigLIPModel
from src.losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from src.utils.seed import set_seed, SEED


# ---------- config ----------

class Config:
    data_root = "data/old/raw"
    train_csv = "data/old/splits/train.csv"
    val_csv = "data/old/splits/val.csv"
    prompts_path = "data/old/prompts_expanded_all.json"

    pretrained_img = "experiments/image_encoder_pretrain/checkpoints/best_encoder.pth"
    pretrained_txt = "experiments/text_encoder_pretrain/checkpoints/best_encoder.pth"

    img_size = 224
    patch_size = 16
    img_dim = 384
    img_depth = 6
    img_heads = 6

    vocab_size = 30522
    max_len = 77
    txt_dim = 256
    txt_depth = 4
    txt_heads = 4
    txt_pool = "mean"
    out_dim = 256

    bs = 32
    accum = 2
    epochs = 40
    img_lr = 5e-6
    txt_lr = 1e-6
    scale_lr = 5e-3
    wd = 0.05
    warmup = 5
    min_lr = 1e-7
    grad_clip = 1.0
    max_scale = 4.6052

    patience = 8
    min_delta = 0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 4

    save_dir = "experiments/siglip"


cfg = Config()
os.makedirs(f"{cfg.save_dir}/checkpoints", exist_ok=True)


# ---------- helpers ----------

def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch]),
    }

def build_ds(csv, mode, tokenizer):
    return OCTDataset(
        csv_path=csv,
        data_root=cfg.data_root,
        prompts_path=cfg.prompts_path,
        transform=get_transforms(mode, cfg.img_size),
        tokenizer=tokenizer,
        mode=mode,
    )

def make_loaders(tokenizer):
    """Creează DataLoader-ele folosind funcțiile globale de mai sus."""
    shared = dict(
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    train_dl = DataLoader(
        build_ds(cfg.train_csv, "train", tokenizer),
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        **shared,
    )

    val_dl = DataLoader(
        build_ds(cfg.val_csv, "eval", tokenizer),
        batch_size=cfg.bs,
        shuffle=False,
        **shared,
    )
    return train_dl, val_dl

def load_pretrained(model):
    """Încarcă greutățile pre-antrenate pentru encodere."""
    mapping = {
        "img_enc": cfg.pretrained_img,
        "txt_enc": cfg.pretrained_txt,
    }
    for attr, path in mapping.items():
        if not os.path.exists(path):
            print(f"  {attr} not found, starting from scratch")
            continue

        weights = torch.load(path, map_location="cpu", weights_only=True)
        result = getattr(model, attr).load_state_dict(weights, strict=False)
        print(f"  {attr}: {path}")
        if result.missing_keys:
            print(f"    missing: {result.missing_keys}")

# ---------- retrieval eval ----------

@torch.no_grad()
def eval_retrieval(model, loader, device):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []

    for batch in tqdm(loader, desc="  Retrieval"):
        with autocast(device, enabled=cfg.amp):
            ie = model.encode_image(batch["images"].to(device))
            te = model.encode_text(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        all_img.append(ie.cpu())
        all_txt.append(te.cpu())
        all_lbl.append(batch["labels"])

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    metrics = {}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            correct = sum(labels[i] in labels[top[i]] for i in range(n))
            metrics[f"{tag}_R@{k}"] = 100.0 * correct / n

    return metrics


# ---------- train / val ----------

def run_train(model, loader, loss_fn, opt, scaler, ep):
    model.train()
    running_loss = 0.0
    sum_i2t, sum_t2i = 0.0, 0.0
    steps = 0

    opt.zero_grad()

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [train]")
    for step, batch in enumerate(pbar):
        imgs = batch["images"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale = model(imgs, ids, mask)
            loss = loss_fn(ie, te, scale) / cfg.accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, cfg.max_scale)

        with torch.no_grad():
            i2t, t2i = contrastive_accuracy(ie.detach(), te.detach())

        running_loss += loss.item() * cfg.accum
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        pbar.set_postfix(
            loss=f"{loss.item() * cfg.accum:.4f}",
            i2t=f"{i2t:.0f}%",
            scale=f"{model.logit_scale.item():.3f}",
        )

    if (step + 1) % cfg.accum != 0:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        with torch.no_grad():
            model.logit_scale.clamp_(0, cfg.max_scale)

    return running_loss / steps, sum_i2t / steps, sum_t2i / steps


@torch.no_grad()
def run_val(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    sum_i2t, sum_t2i = 0.0, 0.0
    steps = 0

    for batch in tqdm(loader, desc="  Val"):
        imgs = batch["images"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale = model(imgs, ids, mask)
            loss = loss_fn(ie, te, scale)

        i2t, t2i = contrastive_accuracy(ie, te)

        running_loss += loss.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

    return running_loss / steps, sum_i2t / steps, sum_t2i / steps


# ---------- plots ----------

def save_plots(hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(ep, hist["train_loss"], label="Train", marker="o", ms=2)
    axes[0, 0].plot(ep, hist["val_loss"], label="Val", marker="o", ms=2)
    axes[0, 0].set(title="Loss", xlabel="Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, hist["I2T_R@1"], label="R@1", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["I2T_R@5"], label="R@5", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["I2T_R@10"], label="R@10", marker="o", ms=2)
    axes[0, 1].set(title="I2T Retrieval", xlabel="Epoch", ylabel="%")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(ep, hist["T2I_R@1"], label="R@1", marker="o", ms=2)
    axes[1, 0].plot(ep, hist["T2I_R@5"], label="R@5", marker="o", ms=2)
    axes[1, 0].plot(ep, hist["T2I_R@10"], label="R@10", marker="o", ms=2)
    axes[1, 0].set(title="T2I Retrieval", xlabel="Epoch", ylabel="%")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, hist["logit_scale"], color="red", marker="o", ms=2)
    axes[1, 1].set(title="Logit Scale", xlabel="Epoch")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/training_curves.png", dpi=150)
    plt.close()


# ---------- main ----------

def main():
    print(f"{'=' * 70}")
    print("  STAGE 3: SigLIP CONTRASTIVE TRAINING")
    print(f"{'=' * 70}")
    print(f"  Image LR={cfg.img_lr} | Text LR={cfg.txt_lr} | Scale LR={cfg.scale_lr}")
    print(f"  Batch={cfg.bs}x{cfg.accum}={cfg.bs * cfg.accum} | Epochs={cfg.epochs}")

    set_seed()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dl, val_dl = make_loaders(tokenizer)
    print(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")

    model = SigLIPModel(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        img_dim=cfg.img_dim,
        img_depth=cfg.img_depth,
        img_heads=cfg.img_heads,
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_len,
        txt_dim=cfg.txt_dim,
        txt_depth=cfg.txt_depth,
        txt_heads=cfg.txt_heads,
        txt_pool=cfg.txt_pool,
        out_dim=cfg.out_dim,
    )

    print("\nLoading pre-trained weights...")
    load_pretrained(model)
    model = model.to(cfg.device)

    loss_fn = SigLIPLoss()

    opt = torch.optim.AdamW([
        {"params": model.img_enc.parameters(), "lr": cfg.img_lr, "name": "img"},
        {"params": model.txt_enc.parameters(), "lr": cfg.txt_lr, "name": "txt"},
        {"params": [model.logit_scale], "lr": cfg.scale_lr, "weight_decay": 0.0, "name": "scale"},
    ], weight_decay=cfg.wd)

    warmup_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    sched = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[cfg.warmup])

    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    hist_keys = [
        "train_loss", "val_loss",
        "train_i2t_acc", "train_t2i_acc",
        "val_i2t_acc", "val_t2i_acc",
        "I2T_R@1", "I2T_R@5", "I2T_R@10",
        "T2I_R@1", "T2I_R@5", "T2I_R@10",
        "logit_scale", "lr",
    ]
    hist = {k: [] for k in hist_keys}

    best_recall = 0.0
    wait = 0

    print(f"\n{'=' * 70}")

    for ep in range(cfg.epochs):
        t_loss, t_i2t, t_t2i = run_train(model, train_dl, loss_fn, opt, scaler, ep)
        v_loss, v_i2t, v_t2i = run_val(model, val_dl, loss_fn)
        ret = eval_retrieval(model, val_dl, cfg.device)
        sched.step()

        cur_lr = opt.param_groups[0]["lr"]
        scale = model.logit_scale.item()
        avg_r1 = (ret["I2T_R@1"] + ret["T2I_R@1"]) / 2

        hist["train_loss"].append(t_loss)
        hist["val_loss"].append(v_loss)
        hist["train_i2t_acc"].append(t_i2t)
        hist["train_t2i_acc"].append(t_t2i)
        hist["val_i2t_acc"].append(v_i2t)
        hist["val_t2i_acc"].append(v_t2i)
        hist["logit_scale"].append(scale)
        hist["lr"].append(cur_lr)
        for k in ["I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10"]:
            hist[k].append(ret[k])

        print(
            f"\nEp {ep + 1}: Loss T={t_loss:.4f} V={v_loss:.4f} | "
            f"I2T R@1={ret['I2T_R@1']:.1f}% R@5={ret['I2T_R@5']:.1f}% | "
            f"T2I R@1={ret['T2I_R@1']:.1f}% R@5={ret['T2I_R@5']:.1f}% | "
            f"Scale={scale:.3f}"
        )

        if avg_r1 > best_recall + cfg.min_delta:
            best_recall = avg_r1
            wait = 0
            print(f"  Best R@1: {best_recall:.1f}%")

            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_recall": best_recall,
                "hist": hist,
            }, f"{cfg.save_dir}/checkpoints/best.pth")

            torch.save(model.img_enc.state_dict(), f"{cfg.save_dir}/checkpoints/best_img_enc.pth")
            torch.save(model.txt_enc.state_dict(), f"{cfg.save_dir}/checkpoints/best_txt_enc.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {ep + 1}")
            break

    torch.save(model.state_dict(), f"{cfg.save_dir}/checkpoints/final.pth")

    os.makedirs("checkpoints", exist_ok=True)
    best_img_path = f"{cfg.save_dir}/checkpoints/best_img_enc.pth"
    best_txt_path = f"{cfg.save_dir}/checkpoints/best_txt_enc.pth"

    if os.path.exists(best_img_path):
        shutil.copy2(best_img_path, "checkpoints/siglip_image_encoder.pth")
        shutil.copy2(best_txt_path, "checkpoints/siglip_text_encoder.pth")
        torch.save(model.state_dict(), "checkpoints/siglip_final.pth")

    pd.DataFrame(hist).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(hist)

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best Avg R@1: {best_recall:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()