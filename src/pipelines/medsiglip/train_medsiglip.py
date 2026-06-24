"""
MedSigLIP v13 — LoRA + BBox Oversampling + lam_cls=1.0 + 27B prompts

Identic cu v12 dar cu prompturi MedGemma 27B (in loc de 4B).
Ablatie: v12 (4B) = 80.0, v13 (27B) = 80.1 — practic identice.

Best results (splits_v3, fara leakage):
  Val:  R@1=86.7% | Cls=75.2% | SevMAE=28.1% | Score=80.1
  Test: R@1=84.0% | Cls=83.2% | SevMAE=24.1% | Score=81.8

v13 changes vs v12:
  - split_json: medgemma_prompts_split_v2_27b.json (era _4b)
  - Restul identic

v12 changes vs v10/v11:
  - lam_cls = 1.0 (era 0.3). cls_head primea prea putin gradient.
  - BBox oversampling in antrenarea normala (WeightedRandomSampler, bbox_weight=3.0).
  - O singura faza de antrenare (fara Phase 2 separata).
  - bs=16, accum=4 (effective 64), LoRA r=16, splits_v3 (fara leakage).
"""

import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.datasets.oct5k_medsiglip import make_loaders, OCT5kDataset, collate_oct5k
from src.losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from src.utils.seed import set_seed


# ---------- config ----------

class Config:
    model_path = "models/medsiglip-448"
    splits_dir = "data/oct5k/splits_v3"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    severity_json = "data/oct5k/severity_scores_v2.json"

    bs = 16
    accum = 4          # effective 64

    epochs = 30
    warmup = 3
    patience = 8
    grad_clip = 1.0
    min_delta = 0.001

    head_lr = 1e-4
    fusion_lr = 5e-5
    wd = 0.01
    min_lr = 1e-7

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_lr = 1.5e-4

    # loss weights
    lam_sev = 0.3
    lam_cls = 1.0

    # bbox oversampling
    bbox_weight = 3.0

    max_scale = 3.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0

    save_dir = "experiments/medsiglip_v13"
    resume = None


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ---------- model ----------

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )

    def forward(self, emb_a, emb_b):
        a, b = emb_a.unsqueeze(1), emb_b.unsqueeze(1)
        attn_a, _ = self.attn_a2b(query=a, key=b, value=b)
        attn_b, _ = self.attn_b2a(query=b, key=a, value=a)
        attn_a, attn_b = attn_a.squeeze(1), attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


class MedSigLIPMultiTask(nn.Module):

    def __init__(self, model_path, n_classes=4):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float32,
        )

        # LoRA pe vision + text (q/k/v/out_proj)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        bb = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        dim = bb.config.vision_config.hidden_size

        # heads pe features NE-normalizate + LayerNorm la intrare
        self.sev_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

        self.fusion = CrossAttentionFusion(dim, heads=4, dropout=0.1)

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"  MedSigLIP v13:")
        print(f"    Total: {n_total:,} | Trainable: {n_train:,} | Emb dim: {dim}")
        print(f"    Effective batch: {cfg.bs} x {cfg.accum} = {cfg.bs * cfg.accum}")
        print(f"    lam_cls={cfg.lam_cls} | bbox_weight={cfg.bbox_weight}")
        self.backbone.print_trainable_parameters()

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return out   # ne-normalizat

    def encode_text(self, input_ids, attention_mask):
        out = self.backbone.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        pooled = self.encode_image(pixel_values)
        img_emb = F.normalize(pooled, p=2, dim=-1)   # normalizat doar pt contrastive
        ea = self.encode_text(ids_a, mask_a)
        eb = self.encode_text(ids_b, mask_b)
        merged = self.fusion(ea, eb)
        sev = self.sev_head(pooled).squeeze(-1).clamp(0, 1)
        cls = self.cls_head(pooled)
        return img_emb, ea, eb, merged, self.logit_scale, sev, cls


# ---------- train loader cu bbox oversampling ----------

def make_train_loader_weighted(proc, cfg):
    train_csv = f"{cfg.splits_dir}/train.csv"

    ds = OCT5kDataset(
        split_csv=train_csv,
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=proc,
        mode="train",
    )

    n_bbox = 0
    if "has_bbox" in ds.df.columns:
        weights = []
        for bb in ds.df["has_bbox"]:
            if bb:
                weights.append(cfg.bbox_weight)
                n_bbox += 1
            else:
                weights.append(1.0)
    else:
        weights = [1.0] * len(ds)

    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)

    print(f"  Train: {len(ds)} imagini ({n_bbox} cu bbox, weight={cfg.bbox_weight}x)")

    return ds, DataLoader(
        ds,
        batch_size=cfg.bs,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_oct5k,
        drop_last=True,
    )


# ---------- optimizer ----------

def make_optimizer(model):
    lora_decay, lora_nodecay = [], []
    fusion_decay, fusion_nodecay = [], []
    head_decay, head_nodecay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "lora" in name.lower():
            target_d, target_nd = lora_decay, lora_nodecay
        elif "fusion" in name:
            target_d, target_nd = fusion_decay, fusion_nodecay
        else:
            target_d, target_nd = head_decay, head_nodecay

        if p.ndim <= 1 or "bias" in name or "norm" in name:
            target_nd.append(p)
        else:
            target_d.append(p)

    groups = [
        {"params": lora_decay,    "lr": cfg.lora_lr,   "weight_decay": cfg.wd,  "name": "lora"},
        {"params": lora_nodecay,  "lr": cfg.lora_lr,   "weight_decay": 0.0,     "name": "lora_nd"},
        {"params": fusion_decay,  "lr": cfg.fusion_lr,  "weight_decay": cfg.wd,  "name": "fusion"},
        {"params": fusion_nodecay,"lr": cfg.fusion_lr,  "weight_decay": 0.0,     "name": "fusion_nd"},
        {"params": head_decay,    "lr": cfg.head_lr,    "weight_decay": cfg.wd,  "name": "heads"},
        {"params": head_nodecay,  "lr": cfg.head_lr,    "weight_decay": 0.0,     "name": "heads_nd"},
    ]
    groups = [g for g in groups if g["params"]]

    print("  Optimizer groups:")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"    {g['name']}: {n:,} params, lr={g['lr']}")

    return torch.optim.AdamW(groups)


# ---------- memory ----------

def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------- eval ----------

@torch.no_grad()
def eval_all(model, loader):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []
    all_sp, all_st = [], []
    all_cp, all_ct = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, te, _, sp, cl = model(pv, ia, ma, ib, mb)

        all_img.append(ie.cpu())
        all_txt.append(te.cpu())
        all_lbl.append(batch["label"])
        all_sp.append(sp.cpu())
        all_st.append(batch["severity"])
        all_cp.append(cl.argmax(1).cpu())
        all_ct.append(batch["label"])

        del pv, ia, ma, ib, mb, ie, ea, eb, te, sp, cl

    clear_mem()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    out = {}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            hit = sum(labels[i] in labels[top[i]] for i in range(n))
            out[f"{tag}_R@{k}"] = 100.0 * hit / n

    sp_pct = torch.cat(all_sp) * 100
    st_pct = torch.cat(all_st) * 100
    out["sev_mae"] = (sp_pct - st_pct).abs().mean().item()

    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    out["cls_acc"] = (cp == ct).float().mean().item() * 100

    return out


# ---------- train ----------

def run_train(model, loader, c_loss, opt, scaler, ep):
    model.train()
    tot_l, tot_c, tot_s, tot_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0

    cls_fn = nn.CrossEntropyLoss()
    sev_fn = nn.SmoothL1Loss()

    opt.zero_grad()

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs}")
    for step, batch in enumerate(pbar):

        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)
        severity = batch["severity"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, merged, scale, sp, cl = model(pv, ia, ma, ib, mb)
            lc_a = c_loss(ie, ea, scale)
            lc_b = c_loss(ie, eb, scale)
            lc_m = c_loss(ie, merged, scale)
            lc = (lc_a + lc_b + lc_m) / 3
            ls = sev_fn(sp, severity)
            lcl = cls_fn(cl, labels)
            loss = lc + cfg.lam_sev * ls + cfg.lam_cls * lcl
            loss_div = loss / cfg.accum

        scaler.scale(loss_div).backward()

        if (step + 1) % cfg.accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, cfg.max_scale)

        with torch.no_grad():
            i2t, t2i = contrastive_accuracy(ie.detach(), merged.detach())

        tot_l += loss.item()
        tot_c += lc.item()
        tot_s += ls.item()
        tot_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        del pv, ia, ma, ib, mb, labels, severity
        del ie, ea, eb, merged, sp, cl, loss, loss_div, lc, lc_a, lc_b, lc_m, ls, lcl

        pbar.set_postfix(
            L=f"{tot_l / steps:.3f}",
            C=f"{tot_c / steps:.3f}",
            S=f"{tot_s / steps:.3f}",
            CL=f"{tot_cl / steps:.3f}",
            i2t=f"{sum_i2t / steps:.0f}%",
        )

    clear_mem()

    return {
        "loss": tot_l / steps, "loss_c": tot_c / steps,
        "loss_s": tot_s / steps, "loss_cl": tot_cl / steps,
        "i2t": sum_i2t / steps, "t2i": sum_t2i / steps,
    }


# ---------- val ----------

@torch.no_grad()
def run_val(model, loader, c_loss):
    model.eval()
    tot_l, tot_c, tot_s, tot_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0

    cls_fn = nn.CrossEntropyLoss()
    sev_fn = nn.SmoothL1Loss()

    for batch in tqdm(loader, desc="  Val", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)
        severity = batch["severity"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, merged, scale, sp, cl = model(pv, ia, ma, ib, mb)
            lc_a = c_loss(ie, ea, scale)
            lc_b = c_loss(ie, eb, scale)
            lc_m = c_loss(ie, merged, scale)
            lc = (lc_a + lc_b + lc_m) / 3
            ls = sev_fn(sp, severity)
            lcl = cls_fn(cl, labels)
            loss = lc + cfg.lam_sev * ls + cfg.lam_cls * lcl

        i2t, t2i = contrastive_accuracy(ie, merged)
        tot_l += loss.item()
        tot_c += lc.item()
        tot_s += ls.item()
        tot_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        del pv, ia, ma, ib, mb, labels, severity
        del ie, ea, eb, merged, sp, cl, loss, lc, lc_a, lc_b, lc_m, ls, lcl

    clear_mem()

    return {
        "loss": tot_l / steps, "loss_c": tot_c / steps,
        "loss_s": tot_s / steps, "loss_cl": tot_cl / steps,
        "i2t": sum_i2t / steps, "t2i": sum_t2i / steps,
    }


# ---------- plots ----------

def save_plots(hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(ep, hist["train_loss"], label="Train", marker="o", ms=2)
    axes[0, 0].plot(ep, hist["val_loss"], label="Val", marker="o", ms=2)
    axes[0, 0].set(title="Total Loss", xlabel="Epoch")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, hist["train_loss_c"], label="Contrastive", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["train_loss_s"], label="Severity", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["train_loss_cl"], label="Classification", marker="o", ms=2)
    axes[0, 1].set(title="Train Loss Breakdown", xlabel="Epoch")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(ep, hist["I2T_R@1"], label="R@1", marker="o", ms=2)
    axes[0, 2].plot(ep, hist["I2T_R@5"], label="R@5", marker="o", ms=2)
    axes[0, 2].plot(ep, hist["I2T_R@10"], label="R@10", marker="o", ms=2)
    axes[0, 2].set(title="I2T Retrieval", xlabel="Epoch", ylabel="%")
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(ep, hist["cls_acc"], marker="o", ms=2, color="green")
    axes[1, 0].set(title="Classification Accuracy", xlabel="Epoch", ylabel="%")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, hist["sev_mae"], marker="o", ms=2, color="orange")
    axes[1, 1].set(title="Severity MAE (%)", xlabel="Epoch")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(ep, hist["logit_scale"], color="red", marker="o", ms=2)
    axes[1, 2].set(title="Logit Scale", xlabel="Epoch")
    axes[1, 2].grid(alpha=0.3)

    plt.suptitle(
        f"MedSigLIP v13 — LoRA r={cfg.lora_r} | lam_cls={cfg.lam_cls} | bbox_weight={cfg.bbox_weight}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/training_curves.png", dpi=150)
    plt.close()


# ---------- main ----------

def main():
    print(f"{'=' * 70}")
    print("  MEDSIGLIP v13 — LoRA + BBox Oversampling + lam_cls=1.0 + 27B prompts")
    print(f"  split_json = {cfg.split_json}")
    print(f"  bs={cfg.bs} x accum={cfg.accum} = {cfg.bs * cfg.accum} effective")
    print(f"  lam_cls={cfg.lam_cls} | lam_sev={cfg.lam_sev} | bbox_weight={cfg.bbox_weight}")
    print(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} lr={cfg.lora_lr}")
    print(f"{'=' * 70}")

    set_seed()

    wandb.init(
        project="licenta-medsiglip",
        name="v13-lora-cls1.0-bbox3x-27b",
        config={
            "version": "v13",
            "model": cfg.model_path,
            "bs_effective": cfg.bs * cfg.accum,
            "epochs": cfg.epochs,
            "prompts": "MedGemma 27B IT",
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_lr": cfg.lora_lr,
            "lam_cls": cfg.lam_cls,
            "lam_sev": cfg.lam_sev,
            "bbox_weight": cfg.bbox_weight,
            "splits": "splits_v3 (patient-grouped)",
            "dataset": "OCT5k",
        }
    )

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    # train loader cu bbox oversampling
    train_ds, train_dl = make_train_loader_weighted(proc, cfg)

    # val/test
    _, val_dl, test_dl = make_loaders(proc, cfg)

    if val_dl is None:
        raise RuntimeError("Val loader missing!")

    nc = train_ds.n_classes
    print(f"  Val: {len(val_dl.dataset)} | Classes: {train_ds.classes}")

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc).to(cfg.device)
    loss_fn = SigLIPLoss()

    print("\n  Optimizer:")
    opt = make_optimizer(model)

    w_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    c_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    sched = SequentialLR(opt, [w_sched, c_sched], milestones=[cfg.warmup])
    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    hist_keys = [
        "train_loss", "val_loss",
        "train_loss_c", "train_loss_s", "train_loss_cl",
        "val_loss_c", "val_loss_s", "val_loss_cl",
        "I2T_R@1", "I2T_R@5", "I2T_R@10",
        "T2I_R@1", "T2I_R@5", "T2I_R@10",
        "cls_acc", "sev_mae", "logit_scale", "lr",
    ]
    hist = {k: [] for k in hist_keys}
    best = 0.0
    wait = 0
    start_ep = 0

    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["epoch"] + 1
        best = ckpt["best_score"]
        wait = ckpt["wait"]
        hist = ckpt["hist"]
        print(f"  Resumed from epoch {start_ep}, best: {best:.1f}")

    # ================================================================
    # antrenare cu bbox oversampling (o singura faza)
    # ================================================================
    for ep in range(start_ep, cfg.epochs):
        t = run_train(model, train_dl, loss_fn, opt, scaler, ep)
        clear_mem()

        v = run_val(model, val_dl, loss_fn)
        clear_mem()

        m = eval_all(model, val_dl)
        clear_mem()

        sched.step()

        scale = model.logit_scale.item()
        avg_r1 = (m["I2T_R@1"] + m["T2I_R@1"]) / 2
        score = 0.5 * avg_r1 + 0.25 * m["cls_acc"] + 0.25 * max(0, 100 - m["sev_mae"])

        hist["train_loss"].append(t["loss"])
        hist["val_loss"].append(v["loss"])
        hist["train_loss_c"].append(t["loss_c"])
        hist["train_loss_s"].append(t["loss_s"])
        hist["train_loss_cl"].append(t["loss_cl"])
        hist["val_loss_c"].append(v["loss_c"])
        hist["val_loss_s"].append(v["loss_s"])
        hist["val_loss_cl"].append(v["loss_cl"])
        hist["cls_acc"].append(m["cls_acc"])
        hist["sev_mae"].append(m["sev_mae"])
        hist["logit_scale"].append(scale)
        hist["lr"].append(opt.param_groups[0]["lr"])
        for k in ["I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10"]:
            hist[k].append(m[k])

        print(
            f"\nEp {ep + 1}: Loss T={t['loss']:.3f} V={v['loss']:.3f} "
            f"[C={t['loss_c']:.3f} S={t['loss_s']:.3f} CL={t['loss_cl']:.3f}]"
        )
        print(
            f"  R@1={avg_r1:.1f}% | Cls={m['cls_acc']:.1f}% | "
            f"SevMAE={m['sev_mae']:.1f}% | Score={score:.1f}"
        )

        wandb.log({
            "epoch": ep + 1,
            "train/loss": t["loss"],
            "train/loss_contrastive": t["loss_c"],
            "train/loss_severity": t["loss_s"],
            "train/loss_classification": t["loss_cl"],
            "val/loss": v["loss"],
            "val/R@1": avg_r1,
            "val/I2T_R@1": m["I2T_R@1"],
            "val/T2I_R@1": m["T2I_R@1"],
            "val/cls_acc": m["cls_acc"],
            "val/sev_mae": m["sev_mae"],
            "val/score": score,
            "logit_scale": scale,
            "lr": opt.param_groups[0]["lr"],
        })

        ckpt = {
            "epoch": ep, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "best_score": best,
            "wait": wait, "hist": hist,
            "num_classes": nc, "classes": train_ds.classes,
            "version": "v13",
        }

        if score > best + cfg.min_delta:
            best = score
            wait = 0
            print(f"  ★ Best: {best:.1f}")
            torch.save(ckpt, f"{cfg.save_dir}/ckpts/best.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        torch.save(ckpt, f"{cfg.save_dir}/ckpts/last.pth")

        if wait >= cfg.patience:
            print(f"  Early stopping la epoch {ep + 1}")
            break

    # ================================================================
    # FINAL
    # ================================================================
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/medsiglip_v13_final.pth")

    pd.DataFrame(hist).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(hist)

    wandb.finish()

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best Score: {best:.1f}")
    print(f"  v13: LoRA + lam_cls=1.0 + bbox oversampling 3x + 27B prompts")
    print(f"  Saved: checkpoints/medsiglip_v13_final.pth")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()