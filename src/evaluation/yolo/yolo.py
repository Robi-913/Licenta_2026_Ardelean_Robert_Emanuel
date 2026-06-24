"""
Comparatie YOLO12s vs MedSigLIP v5 Biomarker Heads v3
pe test set OCT5k (79 imagini cu bbox annotations)

YOLO: bounding boxes -> labels binare
MedSigLIP v3: OCT only + focal loss + unfreeze + threshold optimizat

Rulare:
    python src/evaluation/yolo/yolo.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.seed import set_seed


# ================================================================
# CONFIG
# ================================================================

class Config:
    yolo_ckpt   = "models/yolo12s_oct5k.pt"
    yolo_conf   = 0.25
    yolo_iou    = 0.45

    model_path  = "models/medsiglip-448"
    ms_ckpt     = "experiments/medsiglip_v5/ckpts/best.pth"
    bm_ckpt     = "experiments/biomarker_heads_v3/ckpts/best.pth"

    splits_dir  = "data/oct5k/splits"
    master_json = "data/oct5k/metadata/_master.json"

    out_json    = "experiments/yolo_vs_medsiglip_v3.json"

    bs      = 16
    workers = 0
    device  = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs("experiments", exist_ok=True)


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
# DATASET
# ================================================================

IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


def locate_image(meta):
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


def load_test_bbox_data(splits_dir, master_json):
    test_df  = pd.read_csv(f"{splits_dir}/test.csv")
    bbox_df  = test_df[test_df["has_bbox"] == True].reset_index(drop=True)

    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_index = {m["image_path"]: m for m in metadata}

    samples = []
    for _, row in bbox_df.iterrows():
        path = row["image_path"]
        meta = meta_index.get(path)
        if meta is None:
            continue

        img_path = locate_image(meta)
        if img_path is None:
            continue

        labels = torch.zeros(N_BM)
        seen = set()
        for les in meta.get("lesions", []):
            cls_norm = normalize_class(les.get("class", ""))
            bm_key   = BM_NORMALIZED.get(cls_norm)
            if bm_key and bm_key not in seen:
                labels[BM2IDX[bm_key]] = 1.0
                seen.add(bm_key)

        samples.append({
            "image_path": path,
            "img_path":   img_path,
            "labels":     labels,
        })

    print(f"  Test bbox images: {len(samples)}")
    return samples


# ================================================================
# YOLO
# ================================================================

def eval_yolo(samples, yolo_model):
    print(f"\n  Running YOLO on {len(samples)} images...")
    all_preds, all_labels = [], []

    for s in tqdm(samples, desc="  YOLO"):
        results = yolo_model(s["img_path"], conf=cfg.yolo_conf, iou=cfg.yolo_iou, verbose=False)

        pred = torch.zeros(N_BM)
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id   = int(box.cls.item())
                    cls_norm = normalize_class(r.names.get(cls_id, ""))
                    bm_key   = BM_NORMALIZED.get(cls_norm)
                    if bm_key and bm_key in BM2IDX:
                        pred[BM2IDX[bm_key]] = 1.0

        all_preds.append(pred)
        all_labels.append(s["labels"])

    return torch.stack(all_preds).numpy(), torch.stack(all_labels).numpy()


# ================================================================
# MEDSIGLIP v3 HEADS
# ================================================================

class BiomarkerHeadsV3(nn.Module):
    """v3: OCT only, heads mai mari (512->128->1), backbone partial unfrozen."""

    def __init__(self, backbone, dim, n_bm=9, unfreeze_last_n=2):
        super().__init__()
        self.backbone = backbone

        # freeze tot
        for p in self.backbone.parameters():
            p.requires_grad = False

        # unfreeze ultimele N straturi
        vision_layers = self.backbone.vision_model.encoder.layers
        n_layers = len(vision_layers)
        for i in range(max(0, n_layers - unfreeze_last_n), n_layers):
            for p in vision_layers[i].parameters():
                p.requires_grad = True

        # heads mari
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            for _ in range(n_bm)
        ])

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values):
        img_emb = self.encode_image(pixel_values)
        logits  = torch.cat([h(img_emb) for h in self.heads], dim=-1)
        return logits


class SimpleDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples   = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        px  = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": px["pixel_values"].squeeze(0),
            "labels":       s["labels"],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       torch.stack([b["labels"] for b in batch]),
    }


@torch.no_grad()
def eval_medsiglip(samples, model, proc, thresholds):
    print(f"\n  Running MedSigLIP v3 heads on {len(samples)} images...")
    print(f"  Thresholds: {[round(t, 2) for t in thresholds]}")

    ds     = SimpleDataset(samples, proc)
    loader = DataLoader(
        ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate_fn,
    )

    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="  MedSigLIP"):
        pv = batch["pixel_values"].to(cfg.device)

        with torch.no_grad():
            logits = model(pv)
            probs  = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"])
        del pv, logits, probs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    # aplica thresholds per biomarker
    preds = np.zeros_like(probs)
    for i in range(N_BM):
        preds[:, i] = (probs[:, i] > thresholds[i]).astype(float)

    return preds, labels


# ================================================================
# METRICS
# ================================================================

def compute_metrics(preds, labels, model_name):
    results = {"model": model_name, "per_biomarker": {}}
    f1s = []

    print(f"\n  {model_name}:")
    print(f"  {'Biomarker':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6} {'GT+':>5}")
    print(f"  {'-'*60}")

    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(labels[:, i].sum())
        if n_pos == 0:
            results["per_biomarker"][bm] = {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n_positive": 0,
            }
            continue

        f1   = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec  = recall_score(labels[:, i], preds[:, i], zero_division=0)
        acc  = accuracy_score(labels[:, i], preds[:, i])

        results["per_biomarker"][bm] = {
            "f1":        round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec), 4),
            "accuracy":  round(float(acc * 100), 2),
            "n_positive": n_pos,
        }
        f1s.append(f1)
        print(f"  {bm:<25} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {acc*100:>5.1f}% {n_pos:>5}")

    macro_f1    = float(np.mean(f1s)) if f1s else 0.0
    all_gt      = labels.flatten()
    all_pred    = preds.flatten()
    overall_f1  = f1_score(all_gt, all_pred, zero_division=0)
    overall_acc = accuracy_score(all_gt, all_pred)

    results["macro_f1"]    = round(macro_f1, 4)
    results["overall_f1"]  = round(float(overall_f1), 4)
    results["overall_acc"] = round(float(overall_acc * 100), 2)

    print(f"\n  Macro F1:    {macro_f1:.4f}")
    print(f"  Overall F1:  {overall_f1:.4f}")
    print(f"  Overall Acc: {overall_acc*100:.2f}%")

    return results


def print_comparison(yolo_results, ms_results):
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: YOLO12s vs MedSigLIP v5 Biomarker Heads v3")
    print(f"  Test set: 79 imagini cu bbox | OCT original | threshold optimizat")
    print(f"{'=' * 70}")
    print(f"  {'Biomarker':<25} {'YOLO F1':>8} {'MS F1':>8} {'Winner':>10}")
    print(f"  {'-'*57}")

    yolo_bm = yolo_results["per_biomarker"]
    ms_bm   = ms_results["per_biomarker"]
    yolo_wins = ms_wins = ties = 0

    for bm in BIOMARKERS:
        yf = yolo_bm.get(bm, {}).get("f1", 0.0)
        mf = ms_bm.get(bm, {}).get("f1", 0.0)

        if yf > mf + 0.01:
            winner = "YOLO"
            yolo_wins += 1
        elif mf > yf + 0.01:
            winner = "MedSigLIP"
            ms_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"  {bm:<25} {yf:>8.3f} {mf:>8.3f} {winner:>10}")

    print(f"  {'-'*57}")
    print(f"  {'MACRO F1':<25} {yolo_results['macro_f1']:>8.3f} {ms_results['macro_f1']:>8.3f}")
    print(f"  {'OVERALL F1':<25} {yolo_results['overall_f1']:>8.3f} {ms_results['overall_f1']:>8.3f}")
    print(f"  {'OVERALL ACC':<25} {yolo_results['overall_acc']:>7.2f}% {ms_results['overall_acc']:>7.2f}%")
    print(f"\n  YOLO wins: {yolo_wins} | MedSigLIP wins: {ms_wins} | Ties: {ties}")
    print(f"{'=' * 70}")


# ================================================================
# MAIN
# ================================================================

def main():
    set_seed()

    print("=" * 70)
    print("  YOLO12s vs MedSigLIP v5 Biomarker Heads v3")
    print("  Task: detectie binara prezenta/absenta per biomarker")
    print("  MedSigLIP v3: OCT only + focal loss + unfreeze + threshold opt")
    print("  Dataset: test set OCT5k (79 imagini cu bbox)")
    print("=" * 70)

    samples = load_test_bbox_data(cfg.splits_dir, cfg.master_json)

    # ---- YOLO ----
    print(f"\n  Loading YOLO from {cfg.yolo_ckpt}...")
    yolo_model = YOLO(cfg.yolo_ckpt)
    print(f"  YOLO classes: {yolo_model.names}")

    yolo_preds, gt_labels = eval_yolo(samples, yolo_model)
    yolo_results = compute_metrics(yolo_preds, gt_labels, "YOLO12s")

    del yolo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- MedSigLIP v3 ----
    print(f"\n  Loading MedSigLIP v3 from {cfg.bm_ckpt}...")
    proc     = AutoProcessor.from_pretrained(cfg.model_path)
    backbone = AutoModel.from_pretrained(cfg.model_path, torch_dtype=torch.float32)

    # incarca backbone din v5
    ms_ckpt = torch.load(cfg.ms_ckpt, map_location="cpu", weights_only=False)
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in ms_ckpt["model"].items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(backbone_state, strict=True)

    dim   = backbone.config.vision_config.hidden_size
    model = BiomarkerHeadsV3(backbone, dim, n_bm=N_BM, unfreeze_last_n=2).to(cfg.device)

    # incarca v3 checkpoint (model complet)
    bm_ckpt = torch.load(cfg.bm_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(bm_ckpt["model"])
    model.eval()

    # thresholds optimizate
    thresholds = bm_ckpt.get("thresholds", [0.5] * N_BM)
    thresholds = [float(t) for t in thresholds]
    print(f"  Thresholds: {[round(t, 2) for t in thresholds]}")
    print(f"  Best F1 (val, thr opt): {bm_ckpt.get('best_f1_opt', 'N/A')}")
    print("  MedSigLIP v3 loaded!")

    ms_preds, _ = eval_medsiglip(samples, model, proc, thresholds)
    ms_results  = compute_metrics(ms_preds, gt_labels, "MedSigLIP v5 + Heads v3")

    # ---- comparatie ----
    print_comparison(yolo_results, ms_results)

    # ---- save ----
    output = {
        "dataset":    "OCT5k test set bbox only (79 images)",
        "note":       "MedSigLIP v3: OCT only + focal loss + unfreeze + threshold optimized",
        "thresholds": thresholds,
        "yolo":       yolo_results,
        "medsiglip":  ms_results,
    }
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {cfg.out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()