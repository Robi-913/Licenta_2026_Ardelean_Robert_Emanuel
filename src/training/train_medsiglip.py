"""
Step 3: Fine-tune MedSigLIP contrastiv

Încarcă MedSigLIP pre-antrenat (google/medsiglip-448) și face fine-tune
contrastiv cu imaginile OCT originale + prompturile generate de MedGemma.

Input:
    - models/medsiglip-448/              ← model pre-antrenat
    - data/oct5k/splits/                 ← train.csv, val.csv
    - data/oct5k/medgemma_prompts.json   ← prompturi din Step 2

Output:
    - experiments/medsiglip_pipeline/     ← checkpoints, curves, rapoarte

Rulare:
    python -m src.training.train_medsiglip
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import get_medsiglip_loaders
from src.losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from src.utils.seed import set_seed, SEED


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # model
    model_path = "models/medsiglip-448"

    # date
    splits_dir = "data/oct5k/splits"
    prompts_json = "data/oct5k/medgemma_prompts.json"

    # training
    bs = 16
    epochs = 30
    warmup = 3
    patience = 8
    grad_clip = 1.0
    min_delta = 0.001

    # learning rates diferențiate
    vision_lr = 1e-6
    text_lr = 1e-5
    head_lr = 1e-4
    wd = 0.01
    min_lr = 1e-7

    # logit scale
    max_scale = 4.6052

    # amp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0

    # output
    save_dir = "experiments/medsiglip_pipeline"
    resume = None


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class MedSigLIPContrastive(nn.Module):

    def __init__(self, model_path):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        # raport parametri
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  MedSigLIP: {total:,} params total, {trainable:,} trainable")

    def encode_image(self, pixel_values):
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(outputs, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(outputs, p=2, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_emb = self.encode_image(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.logit_scale


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZER CU LR DIFERENȚIATE
# ═══════════════════════════════════════════════════════════════════════

def build_optimizer(model):
    vision_params = []
    text_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision" in name or "visual" in name:
            vision_params.append(param)
        elif "text" in name:
            text_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": vision_params, "lr": cfg.vision_lr, "name": "vision"},
        {"params": text_params, "lr": cfg.text_lr, "name": "text"},
        {"params": other_params, "lr": cfg.head_lr, "name": "head/scale"},
    ]

    # filtrăm grupuri goale
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    for g in param_groups:
        print(f"    {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    return torch.optim.AdamW(param_groups, weight_decay=cfg.wd)


# ═══════════════════════════════════════════════════════════════════════
# RETRIEVAL EVAL
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_retrieval(model, loader):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []

    for batch in tqdm(loader, desc="  Retrieval", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie = model.encode_image(pv)
            te = model.encode_text(ids, mask)

        all_img.append(ie.cpu())
        all_txt.append(te.cpu())
        all_lbl.append(batch["label"])

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


# ═══════════════════════════════════════════════════════════════════════
# TRAIN / VAL LOOPS
# ═══════════════════════════════════════════════════════════════════════

def run_train(model, loader, loss_fn, opt, scaler, ep):
    model.train()
    running_loss = 0.0
    sum_i2t, sum_t2i = 0.0, 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs} [train]")
    for batch in pbar:
        pv = batch["pixel_values"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        opt.zero_grad()

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale = model(pv, ids, mask)
            loss = loss_fn(ie, te, scale)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            model.logit_scale.clamp_(0, cfg.max_scale)
            i2t, t2i = contrastive_accuracy(ie.detach(), te.detach())

        running_loss += loss.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            i2t=f"{i2t:.0f}%",
            scale=f"{model.logit_scale.item():.3f}",
        )

    return running_loss / steps, sum_i2t / steps, sum_t2i / steps


@torch.no_grad()
def run_val(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    sum_i2t, sum_t2i = 0.0, 0.0
    steps = 0

    for batch in tqdm(loader, desc="  Val", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale = model(pv, ids, mask)
            loss = loss_fn(ie, te, scale)

        i2t, t2i = contrastive_accuracy(ie, te)
        running_loss += loss.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

    return running_loss / steps, sum_i2t / steps, sum_t2i / steps


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"{'=' * 70}")
    print("  STEP 3: FINE-TUNE MEDSIGLIP CONTRASTIVE")
    print(f"{'=' * 70}")
    print(f"  Vision LR={cfg.vision_lr} | Text LR={cfg.text_lr} | Head LR={cfg.head_lr}")
    print(f"  Batch={cfg.bs} | Epochs={cfg.epochs}")

    set_seed()

    # processor pt dataset
    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # data
    loaders = get_medsiglip_loaders(processor, cfg)
    train_dl, val_dl = loaders.get("train"), loaders.get("val")

    if train_dl is None or val_dl is None:
        raise RuntimeError("Train sau val loader lipsește!")

    print(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")

    # model
    model = MedSigLIPContrastive(cfg.model_path).to(cfg.device)
    loss_fn = SigLIPLoss()

    # optimizer
    print("\n  Optimizer param groups:")
    opt = build_optimizer(model)

    # scheduler
    warmup_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    sched = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[cfg.warmup])

    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    # history
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
    start_ep = 0

    # resume
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["epoch"] + 1
        best_recall = ckpt["best_recall"]
        wait = ckpt["wait"]
        hist = ckpt["hist"]
        print(f"  Resumed from epoch {start_ep}, best R@1: {best_recall:.1f}%")

    print(f"\n{'=' * 70}")

    for ep in range(start_ep, cfg.epochs):
        t_loss, t_i2t, t_t2i = run_train(model, train_dl, loss_fn, opt, scaler, ep)
        v_loss, v_i2t, v_t2i = run_val(model, val_dl, loss_fn)
        ret = eval_retrieval(model, val_dl)
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
                "wait": wait,
                "hist": hist,
            }, f"{cfg.save_dir}/ckpts/best.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        # save last
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_recall": best_recall,
            "wait": wait,
            "hist": hist,
        }, f"{cfg.save_dir}/ckpts/last.pth")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {ep + 1}")
            break

    # save final
    torch.save(model.state_dict(), f"{cfg.save_dir}/ckpts/final.pth")
    pd.DataFrame(hist).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(hist)

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best Avg R@1: {best_recall:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()