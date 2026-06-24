"""
Biomarker Heads v5 — Doctor-only + LoRA backbone din v13

Schimbari fata de v4:
  - Backbone din v13 (cu LoRA) in loc de v7 (fara LoRA)
  - Doctor-only (fara YOLO silver labels) — ground truth curat
  - splits_v3 (patient-grouped, fara leakage)
  - Backbone complet inghetat (inclusiv LoRA) — antreneaza doar heads
  - Scos sample weights (nu mai e nevoie, nu mai e YOLO)

Arhitectura:
    OCT -> backbone+LoRA (frozen, din v13) -> emb (1152) -> 9 heads (trainable)

Rulare:
    python -m src.pipelines.medsiglip.train_biomarker_v5
"""

import os
import sys
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.seed import set_seed


# ================================================================
# CONFIG
# ================================================================

class Config:
    # v5: backbone din v13 (LoRA)
    backbone_ckpt = "experiments/medsiglip_v13/ckpts/best.pth"
    model_path    = "models/medsiglip-448"

    # v5: splits_v3 (patient-grouped)
    splits_dir    = "data/oct5k/splits_v3"
    master_json   = "data/oct5k/metadata_v2/_master.json"

    # LoRA config — IDENTIC cu v13
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    bs        = 16
    epochs    = 100
    lr_heads  = 5e-4
    wd        = 0.01
    grad_clip = 1.0

    # focal loss
    focal_alpha = 0.25
    focal_gamma = 2.0

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    amp     = torch.cuda.is_available()
    workers = 0

    save_dir = "experiments/biomarker_heads_v5"


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ================================================================
# BIOMARKERI
# ================================================================

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]
N_BM = len(BIOMARKERS)
BM2IDX = {b: i for i, b in enumerate(BIOMARKERS)}


def normalize_class(cls: str) -> str:
    return cls.lower().replace(" ", "").replace("_", "")

BM_NORMALIZED = {normalize_class(b): b for b in BIOMARKERS}


# ================================================================
# DATASET — Doctor-only
# ================================================================

IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]

def locate_oct(meta):
    disk = meta.get("image_disk_path", "")
    if disk and Path(disk).exists():
        return str(disk)
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in IMG_DIRS:
        full = Path(base) / rel
        if full.exists():
            return str(full)
        for ext in [".png", ".jpeg", ".jpg"]:
            if full.with_suffix(ext).exists():
                return str(full.with_suffix(ext))
    return None


class BiomarkerDatasetV5(Dataset):
    """Doctor-only: doar imagini cu bbox_source == 'doctor'."""

    def __init__(self, image_paths, metadata_dict, processor, mode="train"):
        self.processor = processor
        self.mode = mode
        self.samples = []

        for path in image_paths:
            meta = metadata_dict.get(path)
            if meta is None or not meta.get("has_bounding_boxes"):
                continue
            if meta.get("bbox_source", "doctor") != "doctor":
                continue

            oct_path = locate_oct(meta)
            if oct_path is None:
                continue

            labels = torch.zeros(N_BM)
            seen = set()
            for les in meta.get("lesions", []):
                cls_norm = normalize_class(les.get("class", ""))
                bm_key = BM_NORMALIZED.get(cls_norm)
                if bm_key and bm_key not in seen:
                    labels[BM2IDX[bm_key]] = 1.0
                    seen.add(bm_key)

            self.samples.append({
                "oct_path": oct_path,
                "labels": labels,
                "image_path": path,
            })

        print(f"  BiomarkerDatasetV5 [{mode}]: {len(self.samples)} imagini (doctor-only)")

        if self.samples:
            all_labels = torch.stack([s["labels"] for s in self.samples])
            print(f"  Pozitive per biomarker:")
            for i, bm in enumerate(BIOMARKERS):
                n_pos = int(all_labels[:, i].sum())
                print(f"    {bm:<25}: {n_pos}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["oct_path"]).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        if self.mode == "train":
            from torchvision import transforms
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(5),
            ])
            img = aug(img)

        px = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": px["pixel_values"].squeeze(0),
            "labels": s["labels"],
        }


def collate_bm(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction="none")
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ================================================================
# MODEL
# ================================================================

class BiomarkerHeadsV5(nn.Module):
    """Backbone+LoRA complet inghetat, doar 9 heads trainable."""

    def __init__(self, backbone, dim, n_bm=9):
        super().__init__()
        self.backbone = backbone

        # ingheata TOT (inclusiv LoRA)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            for _ in range(n_bm)
        ])

        n_heads = sum(p.numel() for p in self.heads.parameters())
        print(f"  BiomarkerHeadsV5 (doctor-only, LoRA backbone frozen):")
        print(f"    Backbone: frozen (inclusiv LoRA din v13)")
        print(f"    9 heads: {n_heads:,} params trainable")

    def encode_image(self, pixel_values):
        with torch.no_grad():
            out = self.backbone.get_image_features(pixel_values=pixel_values)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state[:, 0]
        return out  # ne-normalizat, pt heads

    def forward(self, pixel_values):
        img_feat = self.encode_image(pixel_values)
        logits = torch.cat([h(img_feat) for h in self.heads], dim=-1)
        return logits


# ================================================================
# CLASS WEIGHTS
# ================================================================

def compute_class_weights(samples):
    all_labels = torch.stack([s["labels"] for s in samples])
    n = len(samples)
    weights = []
    print("\n  Class weights:")
    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(all_labels[:, i].sum())
        n_neg = n - n_pos
        w = n_neg / n_pos if n_pos > 0 else 1.0
        weights.append(w)
        print(f"    {bm:<25}: pos={n_pos}, neg={n_neg}, w={w:.1f}")
    return torch.tensor(weights, dtype=torch.float32)


# ================================================================
# THRESHOLD OPTIMIZATION
# ================================================================

@torch.no_grad()
def optimize_thresholds(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(cfg.device)
        with autocast(cfg.device, enabled=cfg.amp):
            logits = model(pv)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    thresholds = []
    print("\n  Threshold optimization:")
    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(labels[:, i].sum())
        if n_pos == 0:
            thresholds.append(0.5)
            continue
        best_f1, best_thr = 0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(labels[:, i], (probs[:, i] > thr).astype(float), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(round(best_thr, 2))
        print(f"    {bm:<25}: thr={best_thr:.2f} -> F1={best_f1:.3f}")
    return thresholds


# ================================================================
# EVAL
# ================================================================

@torch.no_grad()
def evaluate(model, loader, thresholds=None):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        with autocast(cfg.device, enabled=cfg.amp):
            logits = model(pv)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    if thresholds is None:
        thresholds = [0.5] * N_BM

    results = {}
    f1s = []
    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(labels[:, i].sum())
        if n_pos == 0:
            results[bm] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_pos": 0}
            continue
        preds = (probs[:, i] > thresholds[i]).astype(float)
        f1 = f1_score(labels[:, i], preds, zero_division=0)
        prec = precision_score(labels[:, i], preds, zero_division=0)
        rec = recall_score(labels[:, i], preds, zero_division=0)
        results[bm] = {"f1": round(float(f1), 4), "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4), "n_pos": n_pos}
        f1s.append(f1)

    results["macro_f1"] = round(float(np.mean(f1s)) if f1s else 0.0, 4)
    return results


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("  BIOMARKER HEADS v5 — Doctor-only + LoRA backbone v13")
    print(f"  Epochs: {cfg.epochs} | LR={cfg.lr_heads}")
    print("=" * 60)

    set_seed()

    with open(cfg.master_json, "r", encoding="utf-8") as f:
        master = json.load(f)
    metadata_dict = {m["image_path"]: m for m in master}

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    # loaders — doctor-only
    train_csv = pd.read_csv(f"{cfg.splits_dir}/train.csv")
    val_csv = pd.read_csv(f"{cfg.splits_dir}/val.csv")

    train_paths = train_csv[train_csv["has_bbox"] == True]["image_path"].tolist()
    val_paths = val_csv[val_csv["has_bbox"] == True]["image_path"].tolist()

    train_ds = BiomarkerDatasetV5(train_paths, metadata_dict, proc, mode="train")
    val_ds = BiomarkerDatasetV5(val_paths, metadata_dict, proc, mode="eval")

    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True,
                          num_workers=cfg.workers, pin_memory=True, collate_fn=collate_bm)
    val_dl = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_bm)

    if len(train_ds.samples) == 0:
        raise RuntimeError("Nu s-au gasit imagini doctor cu bbox!")

    pos_weights = compute_class_weights(train_ds.samples)
    loss_fn = FocalLossWithLogits(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma,
                                  pos_weight=pos_weights.to(cfg.device))

    # --- Load backbone cu LoRA din v13 ---
    print(f"\n  Loading backbone + LoRA from {cfg.backbone_ckpt}...")
    backbone = AutoModel.from_pretrained(cfg.model_path, torch_dtype=torch.float32)
    backbone = get_peft_model(backbone, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none",
    ))

    ckpt = torch.load(cfg.backbone_ckpt, map_location="cpu", weights_only=False)
    bb_state = {
        k.replace("backbone.", ""): v
        for k, v in ckpt["model"].items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(bb_state, strict=True)
    print("  Backbone + LoRA loaded!")

    bb = backbone.base_model.model if hasattr(backbone, "base_model") else backbone
    dim = bb.config.vision_config.hidden_size

    model = BiomarkerHeadsV5(backbone, dim, n_bm=N_BM).to(cfg.device)

    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=cfg.lr_heads, weight_decay=cfg.wd)
    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    best_f1 = 0.0
    patience = 20
    wait = 0

    print(f"\n{'=' * 60}")
    for ep in range(cfg.epochs):
        model.train()
        model.backbone.eval()  # backbone frozen, eval mode
        tot_loss, steps = 0, 0

        for batch in tqdm(train_dl, desc=f"  Ep {ep+1}/{cfg.epochs}", leave=False):
            pv = batch["pixel_values"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            with autocast(cfg.device, enabled=cfg.amp):
                logits = model(pv)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            tot_loss += loss.item()
            steps += 1

        metrics = evaluate(model, val_dl)
        macro_f1 = metrics["macro_f1"]

        marker = ""
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            wait = 0
            torch.save({
                "epoch": ep, "model": model.state_dict(),
                "best_f1": best_f1, "metrics": metrics,
                "biomarkers": BIOMARKERS, "version": "v5",
            }, f"{cfg.save_dir}/ckpts/best.pth")
            marker = f"  ★ Best: {best_f1:.4f}"
        else:
            wait += 1

        print(f"  Ep {ep+1}: Loss={tot_loss/steps:.4f} | Macro F1={macro_f1:.4f}{marker}")
        if wait >= patience:
            print(f"  Early stopping la epoch {ep+1}")
            break

    # threshold optimization
    print(f"\n{'=' * 60}")
    print("  THRESHOLD OPTIMIZATION...")
    best_ckpt = torch.load(f"{cfg.save_dir}/ckpts/best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    thresholds = optimize_thresholds(model, val_dl)
    metrics_opt = evaluate(model, val_dl, thresholds)

    print(f"\n  Macro F1 (thr=0.5):      {best_f1:.4f}")
    print(f"  Macro F1 (thr optimizat): {metrics_opt['macro_f1']:.4f}")

    for bm in BIOMARKERS:
        m = metrics_opt[bm]
        if m["n_pos"] > 0:
            print(f"  {bm:<25}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")

    torch.save({
        "epoch": best_ckpt["epoch"], "model": model.state_dict(),
        "best_f1": best_f1, "best_f1_opt": metrics_opt["macro_f1"],
        "thresholds": thresholds, "metrics": metrics_opt,
        "biomarkers": BIOMARKERS, "version": "v5",
    }, f"{cfg.save_dir}/ckpts/best.pth")

    print(f"\n{'=' * 60}")
    print(f"  DONE! Macro F1={metrics_opt['macro_f1']:.4f}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Saved: {cfg.save_dir}/ckpts/best.pth")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()