"""
yolo_vs_medsiglip.py — Comparatie YOLO12s vs MedSigLIP v13 Biomarker Heads v5

Task: detectie binara prezenta/absenta per biomarker
Dataset: test set OCT5k — imagini cu bbox annotations

YOLO:        bounding boxes -> labels binare per clasa
MedSigLIP:   backbone v13 frozen + heads v5 + threshold optimizat

Rulare:
    python src/evaluation/yolo/yolo_vs_medsiglip.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.model.medsiglip import BiomarkerHeadsV5, MedSigLIPMultiTask
from src.utils.seed import set_seed


# CONFIG

class Config:
    yolo_ckpt   = "models/yoloe_oct5k.pt"
    yolo_conf   = 0.25
    yolo_iou    = 0.45

    model_path  = "models/medsiglip-448"
    ms_ckpt     = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"
    bm_ckpt     = "experiments/biomarker_heads_v5/ckpts/best.pth"

    splits_dir  = "data/OCT5k/splits_biomk"
    master_json = "data/oct5k/metadata_v2/_master.json"
    out_json    = "experiments/yolo_vs_medsiglip_v5.json"

    batch_size = 16
    workers    = 0
    device     = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs("experiments", exist_ok=True)


# BIOMARKERI

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]
N_BM    = len(BIOMARKERS)
BM2IDX  = {bm: i for i, bm in enumerate(BIOMARKERS)}
_BM_NORM = {bm.lower().replace(" ", "").replace("_", ""): bm for bm in BIOMARKERS}

def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


# DATA LOADING

_IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]

def _locate_image(meta: dict) -> str | None:
    disk = meta.get("image_disk_path", "")
    if disk and Path(disk).exists():
        return str(disk)
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in _IMG_DIRS:
        candidate = Path(base) / rel
        if candidate.exists():
            return str(candidate)
        for ext in [".png", ".jpeg", ".jpg"]:
            with_ext = candidate.with_suffix(ext)
            if with_ext.exists():
                return str(with_ext)
    return None


def load_test_samples(splits_dir: str, master_json: str) -> list[dict]:
    """
    Incarca imaginile din test split care au bbox annotations.
    Fiecare sample are img_path si un vector binar de labels per biomarker.
    """
    test_df  = pd.read_csv(f"{splits_dir}/test.csv")
    bbox_df  = test_df[test_df["has_bbox"] == True].reset_index(drop=True)

    with open(master_json, "r", encoding="utf-8") as f:
        meta_index = {m["image_path"]: m for m in json.load(f)}

    samples = []
    for _, row in bbox_df.iterrows():
        meta = meta_index.get(row["image_path"])
        if meta is None:
            continue
        img_path = _locate_image(meta)
        if img_path is None:
            continue

        labels = torch.zeros(N_BM)
        seen   = set()
        for les in meta.get("lesions", []):
            bm_key = _BM_NORM.get(_normalize(les.get("class", "")))
            if bm_key and bm_key not in seen:
                labels[BM2IDX[bm_key]] = 1.0
                seen.add(bm_key)

        samples.append({"image_path": row["image_path"], "img_path": img_path, "labels": labels})

    print(f"  Test bbox images: {len(samples)}")
    return samples


# DATASET SIMPLU PT MEDSIGLIP

class BboxDataset(Dataset):
    """Dataset minimal — incarca imaginea si o proceseaza pt backbone."""

    def __init__(self, samples: list[dict], processor: AutoProcessor):
        self.samples   = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # reduce speckle OCT
        px  = self.processor(images=img, return_tensors="pt")
        return {"pixel_values": px["pixel_values"].squeeze(0), "labels": s["labels"]}


def _collate(batch: list) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       torch.stack([b["labels"]       for b in batch]),
    }


# EVALUARE YOLO

def eval_yolo(samples: list[dict], yolo_model: YOLO) -> tuple[np.ndarray, np.ndarray]:
    """
    Ruleaza YOLO pe fiecare imagine si converteste bounding box-urile
    in predictii binare per biomarker (prezent/absent).
    """
    print(f"\n  YOLO inference pe {len(samples)} imagini...")
    all_preds, all_labels = [], []

    for s in tqdm(samples, desc="  YOLO"):

        results = yolo_model(s["img_path"], conf=cfg.yolo_conf, iou=cfg.yolo_iou, verbose=False)
        pred = torch.zeros(N_BM)
        if results and results[0].boxes is not None:
            r = results[0]
            for box in r.boxes:
                bm_key = _BM_NORM.get(_normalize(r.names.get(int(box.cls.item()), "")))
                if bm_key:
                    pred[BM2IDX[bm_key]] = 1.0

        all_preds.append(pred)
        all_labels.append(s["labels"])

    return torch.stack(all_preds).numpy(), torch.stack(all_labels).numpy()


# EVALUARE MEDSIGLIP HEADS V5

def load_biomarker_model() -> tuple[BiomarkerHeadsV5, list[float]]:
    bm_ckpt = torch.load(cfg.bm_ckpt, map_location="cpu", weights_only=False)

    # Autodetect cls_hidden din checkpoint
    cls_hidden = 256
    for key in ["backbone.classification_head.1.weight"]:
        if key in bm_ckpt["model"]:
            cls_hidden = bm_ckpt["model"][key].shape[0]
            break

    base_model = MedSigLIPMultiTask(cfg.model_path, cls_hidden=cls_hidden)
    bm_model = BiomarkerHeadsV5(backbone=base_model, n_biomarkers=N_BM)
    bm_model.load_state_dict(bm_ckpt["model"], strict=False)
    bm_model = bm_model.to(cfg.device).eval()

    thresholds = [float(t) for t in bm_ckpt.get("thresholds", [0.5] * N_BM)]
    print(f"  Thresholds: {[round(t, 2) for t in thresholds]}")
    return bm_model, thresholds


@torch.no_grad()
def eval_medsiglip(
    samples: list[dict],
    model: BiomarkerHeadsV5,
    processor: AutoProcessor,
    thresholds: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n  MedSigLIP inference pe {len(samples)} imagini...")

    loader = DataLoader(
        BboxDataset(samples, processor),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, collate_fn=_collate,
    )

    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="  MedSigLIP"):
        pv    = batch["pixel_values"].to(cfg.device)
        probs = torch.sigmoid(model(pv))
        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"])

    probs_np = torch.cat(all_probs).numpy()
    labels   = torch.cat(all_labels).numpy()

    # Aplicam threshold optimizat per biomarker
    preds = np.stack([
        (probs_np[:, i] > thresholds[i]).astype(float)
        for i in range(N_BM)
    ], axis=1)

    return preds, labels


# METRICI

def compute_metrics(preds: np.ndarray, labels: np.ndarray, model_name: str) -> dict:
    """Calculeaza F1/Precision/Recall/Accuracy per biomarker si macro F1."""
    results = {"model": model_name, "per_biomarker": {}}
    f1_scores = []

    print(f"\n  {model_name}:")
    print(f"  {'Biomarker':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6} {'GT+':>5}")
    print(f"  {'-' * 60}")

    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(labels[:, i].sum())
        if n_pos == 0:
            results["per_biomarker"][bm] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n_positive": 0}
            continue

        f1   = f1_score(labels[:, i], preds[:, i], zero_division=0)
        prec = precision_score(labels[:, i], preds[:, i], zero_division=0)
        rec  = recall_score(labels[:, i], preds[:, i], zero_division=0)
        acc  = accuracy_score(labels[:, i], preds[:, i])

        results["per_biomarker"][bm] = {
            "f1": round(float(f1), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "accuracy": round(float(acc * 100), 2),
            "n_positive": n_pos,
        }
        f1_scores.append(f1)
        print(f"  {bm:<25} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {acc*100:>5.1f}% {n_pos:>5}")

    macro_f1   = float(np.mean(f1_scores)) if f1_scores else 0.0
    overall_f1 = f1_score(labels.flatten(), preds.flatten(), zero_division=0)
    overall_acc = accuracy_score(labels.flatten(), preds.flatten())

    results["macro_f1"]    = round(macro_f1, 4)
    results["overall_f1"]  = round(float(overall_f1), 4)
    results["overall_acc"] = round(float(overall_acc * 100), 2)

    print(f"\n  Macro F1: {macro_f1:.4f} | Overall F1: {overall_f1:.4f} | Acc: {overall_acc*100:.2f}%")
    return results


def print_comparison(yolo_res: dict, ms_res: dict) -> None:
    """Tabel comparativ per biomarker cu winner per rand."""
    print(f"\n  COMPARISON: YOLO12s vs MedSigLIP v13 Biomarker Heads v5")
    print(f"  {'Biomarker':<25} {'YOLO F1':>8} {'MS F1':>8} {'Winner':>12}")
    print(f"  {'-' * 57}")

    yolo_wins = ms_wins = ties = 0
    for bm in BIOMARKERS:
        yf = yolo_res["per_biomarker"].get(bm, {}).get("f1", 0.0)
        mf = ms_res["per_biomarker"].get(bm, {}).get("f1", 0.0)
        if   yf > mf + 0.01: winner = "YOLO";      yolo_wins += 1
        elif mf > yf + 0.01: winner = "MedSigLIP"; ms_wins   += 1
        else:                 winner = "Tie";       ties      += 1
        print(f"  {bm:<25} {yf:>8.3f} {mf:>8.3f} {winner:>12}")

    print(f"  {'-' * 57}")
    print(f"  {'MACRO F1':<25} {yolo_res['macro_f1']:>8.3f} {ms_res['macro_f1']:>8.3f}")
    print(f"  {'OVERALL F1':<25} {yolo_res['overall_f1']:>8.3f} {ms_res['overall_f1']:>8.3f}")
    print(f"  {'OVERALL ACC':<25} {yolo_res['overall_acc']:>7.2f}% {ms_res['overall_acc']:>7.2f}%")
    print(f"\n  YOLO wins: {yolo_wins} | MedSigLIP wins: {ms_wins} | Ties: {ties}")


# MAIN

def main():
    set_seed()

    print("  YOLO12s vs MedSigLIP v13 Biomarker Heads v5")
    print("  Task: detectie binara prezenta/absenta per biomarker")
    print("  Dataset: test set OCT5k — imagini cu bbox")

    samples = load_test_samples(cfg.splits_dir, cfg.master_json)

    # YOLO
    print(f"\n  Loading YOLO din {cfg.yolo_ckpt}...")
    yolo_model = YOLO(cfg.yolo_ckpt)
    print(f"  YOLO classes: {yolo_model.names}")

    yolo_preds, gt_labels = eval_yolo(samples, yolo_model)
    yolo_results          = compute_metrics(yolo_preds, gt_labels, "YOLO12s")
    del yolo_model

    # MedSigLIP
    processor         = AutoProcessor.from_pretrained(cfg.model_path)
    bm_model, thrs    = load_biomarker_model()
    ms_preds, _       = eval_medsiglip(samples, bm_model, processor, thrs)
    ms_results        = compute_metrics(ms_preds, gt_labels, "MedSigLIP v13 + Heads v5")

    print_comparison(yolo_results, ms_results)

    output = {
        "dataset":    "OCT5k test set bbox only",
        "thresholds": thrs,
        "yolo":       yolo_results,
        "medsiglip":  ms_results,
    }
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {cfg.out_json}")


if __name__ == "__main__":
    main()