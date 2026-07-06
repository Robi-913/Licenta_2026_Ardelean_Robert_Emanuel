"""
Retrieval Analysis Detaliata - MedSigLIP v13

Pe test set:
  1. R@1, R@2, R@3 per boala
  2. Confusion matrix retrieval
  3. Failure analysis

Rulare:
    python -m src.evaluation.retrieval_analysis
"""

import json
import os
import sys
import gc
from collections import defaultdict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed



# CONFIG


class Config:
    model_path  = "models/medsiglip-448"
    ckpt_path   = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"

    splits_dir  = "data/oct5k/splits_v3"
    split_json  = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    sev_json    = "data/oct5k/severity_scores_v2.json"

    out_json    = "experiments/retrieval_analysis_v13.json"
    fig_dir     = "experiments/figures/retrieval_analysis_v13"

    bs      = 8
    workers = 0
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    amp     = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.fig_dir, exist_ok=True)

# EXTRACT EMBEDDINGS

def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _detect_cls_head(state_dict):
    for k, v in state_dict.items():
        if k in ["cls_head.1.weight", "classification_head.1.weight"]:
            return v.shape[0] if len(v.shape) > 1 else v.shape[0]
    return 256


@torch.no_grad()
def extract_all(model, loader, dataset):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []
    all_paths, all_diseases = [], []

    for batch_idx, batch in enumerate(tqdm(loader, desc="  Extracting embeddings")):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie = model.encode_image(pv)
            ea = model.encode_text(ia, ma)
            eb = model.encode_text(ib, mb)
            merged = model.fusion(ea, eb)

        all_img.append(ie.cpu())
        all_txt.append(merged.cpu())
        all_lbl.append(batch["label"])

        start = batch_idx * cfg.bs
        end   = min(start + cfg.bs, len(dataset))
        for i in range(end - start):
            row = dataset.df.iloc[start + i]
            all_paths.append(row["image_path"])
            all_diseases.append(row["disease"])

        del pv, ia, ma, ib, mb, ie, ea, eb, merged

    clear_mem()

    return {
        "img_emb": torch.cat(all_img).float(),
        "txt_emb": torch.cat(all_txt).float(),
        "labels": torch.cat(all_lbl),
        "paths": all_paths,
        "diseases": all_diseases,
    }


# RETRIEVAL R@K PER DISEASE


def compute_retrieval_per_disease(img_emb, txt_emb, labels, diseases, classes):
    n = len(labels)
    results = {"overall": {}, "per_class": {}}

    chunk = 256
    sim_i2t_topk = {"indices": [], "values": []}
    sim_t2i_topk = {"indices": [], "values": []}

    print("  Computing similarities (chunked)...")
    for start in tqdm(range(0, n, chunk), desc="  I2T sim", leave=False):
        end = min(start + chunk, n)
        chunk_sim = img_emb[start:end] @ txt_emb.T
        vals, inds = chunk_sim.topk(10, dim=1)
        sim_i2t_topk["indices"].append(inds)
        sim_i2t_topk["values"].append(vals)

    for start in tqdm(range(0, n, chunk), desc="  T2I sim", leave=False):
        end = min(start + chunk, n)
        chunk_sim = txt_emb[start:end] @ img_emb.T
        vals, inds = chunk_sim.topk(10, dim=1)
        sim_t2i_topk["indices"].append(inds)
        sim_t2i_topk["values"].append(vals)

    i2t_top = torch.cat(sim_i2t_topk["indices"], dim=0)
    t2i_top = torch.cat(sim_t2i_topk["indices"], dim=0)

    for tag, top in [("I2T", i2t_top), ("T2I", t2i_top)]:
        for k in [1, 2, 3, 5, 10]:
            correct = 0
            for i in range(n):
                if labels[i] in labels[top[i, :k]]:
                    correct += 1
            results["overall"][f"{tag}_R@{k}"] = round(100.0 * correct / n, 2)

    for k in [1, 2, 3, 5, 10]:
        avg = (results["overall"][f"I2T_R@{k}"] + results["overall"][f"T2I_R@{k}"]) / 2
        results["overall"][f"Avg_R@{k}"] = round(avg, 2)

    for cls_idx, cls_name in enumerate(classes):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        indices = torch.where(mask)[0]
        cls_n = len(indices)
        cls_results = {"count": cls_n}
        for tag, top in [("I2T", i2t_top), ("T2I", t2i_top)]:
            for k in [1, 2, 3, 5, 10]:
                correct = 0
                for idx in indices:
                    if labels[idx] in labels[top[idx, :k]]:
                        correct += 1
                cls_results[f"{tag}_R@{k}"] = round(100.0 * correct / cls_n, 2)
        for k in [1, 2, 3, 5, 10]:
            avg = (cls_results[f"I2T_R@{k}"] + cls_results[f"T2I_R@{k}"]) / 2
            cls_results[f"Avg_R@{k}"] = round(avg, 2)
        results["per_class"][cls_name] = cls_results

    return results, i2t_top, t2i_top



# CONFUSION MATRIX RETRIEVAL


def compute_retrieval_confusion(i2t_top, labels, classes):
    n = len(labels)
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for i in range(n):
        true_cls = labels[i].item()
        retrieved_cls = labels[i2t_top[i, 0]].item()
        cm[true_cls, retrieved_cls] += 1
    return cm



# FAILURE ANALYSIS


def analyze_failures(i2t_top, labels, diseases, paths, classes):
    n = len(labels)
    failures = {"r1_miss_r2_hit": [], "r1_miss_r3_hit": [], "r1_r2_r3_miss": []}
    per_class_stats = {cls: {"total": 0, "r1_hit": 0, "r2_hit": 0, "r3_hit": 0} for cls in classes}

    for i in range(n):
        true_cls = labels[i].item()
        cls_name = classes[true_cls]
        per_class_stats[cls_name]["total"] += 1
        top_labels = [labels[t].item() for t in i2t_top[i, :10]]
        r1_hit = top_labels[0] == true_cls
        r2_hit = true_cls in top_labels[:2]
        r3_hit = true_cls in top_labels[:3]
        if r1_hit: per_class_stats[cls_name]["r1_hit"] += 1
        if r2_hit: per_class_stats[cls_name]["r2_hit"] += 1
        if r3_hit: per_class_stats[cls_name]["r3_hit"] += 1
        if not r1_hit:
            retrieved_cls = classes[top_labels[0]]
            info = {"index": i, "true_class": cls_name, "retrieved_class": retrieved_cls,
                    "path": paths[i], "top3_classes": [classes[c] for c in top_labels[:3]]}
            if r2_hit: failures["r1_miss_r2_hit"].append(info)
            elif r3_hit: failures["r1_miss_r3_hit"].append(info)
            else: failures["r1_r2_r3_miss"].append(info)

    return failures, per_class_stats



# PLOTS


def plot_retrieval_per_class(results, classes):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for k_idx, k in enumerate([1, 2, 3]):
        vals = [results["per_class"].get(cls, {}).get(f"Avg_R@{k}", 0) for cls in classes]
        bars = ax.bar(x + k_idx * width, vals, width, label=f"R@{k}", color=colors[k_idx])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Retrieval %", fontsize=12)
    ax.set_title("MedSigLIP v13 - Avg R@1, R@2, R@3 per Disease", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/retrieval_per_class.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/retrieval_per_class.png")


def plot_confusion_matrix(cm, classes):
    cm_norm = cm.astype(float)
    for i in range(len(classes)):
        row_sum = cm_norm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm_norm[i] / row_sum * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Retrieval Confusion (counts)", fontsize=13)
    axes[0].set_ylabel("True Class"); axes[0].set_xlabel("Retrieved Class")
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Reds", xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Retrieval Confusion (%)", fontsize=13)
    axes[1].set_ylabel("True Class"); axes[1].set_xlabel("Retrieved Class")
    plt.suptitle("MedSigLIP v13 - Retrieval Confusion (I2T top-1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/retrieval_confusion.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/retrieval_confusion.png")


def plot_failure_breakdown(per_class_stats, classes):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.6
    r1_hits, r2_recovery, r3_recovery, misses = [], [], [], []
    for cls in classes:
        s = per_class_stats[cls]
        total = s["total"]
        r1 = s["r1_hit"]
        r2 = s["r2_hit"] - r1
        r3 = s["r3_hit"] - s["r2_hit"]
        miss = total - s["r3_hit"]
        r1_hits.append(r1 / total * 100 if total > 0 else 0)
        r2_recovery.append(r2 / total * 100 if total > 0 else 0)
        r3_recovery.append(r3 / total * 100 if total > 0 else 0)
        misses.append(miss / total * 100 if total > 0 else 0)
    r1_hits, r2_recovery, r3_recovery, misses = map(np.array, [r1_hits, r2_recovery, r3_recovery, misses])
    ax.bar(x, r1_hits, width, label="R@1 Hit", color="#2ecc71")
    ax.bar(x, r2_recovery, width, bottom=r1_hits, label="R@2 Recovery", color="#3498db")
    ax.bar(x, r3_recovery, width, bottom=r1_hits + r2_recovery, label="R@3 Recovery", color="#f39c12")
    ax.bar(x, misses, width, bottom=r1_hits + r2_recovery + r3_recovery, label="Miss", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("% of images", fontsize=12)
    ax.set_title("MedSigLIP v13 - Retrieval Hit/Recovery/Miss per Disease", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/failure_breakdown.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/failure_breakdown.png")



# MAIN


def main():
    set_seed()

    print("=" * 70)
    print("  RETRIEVAL ANALYSIS - MedSigLIP v13")
    print("  R@1, R@2, R@3 per boala | Confusion | Failure analysis")
    print("=" * 70)

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # --- REMAPPING CHEI VECHI -> NOI ---
    remapped = {
        k.replace("sev_head.", "severity_head.")
        .replace("cls_head.", "classification_head.")
        .replace("fusion.attn_a2b.", "fusion.attn_a_to_b.")
        .replace("fusion.attn_b2a.", "fusion.attn_b_to_a."): v
        for k, v in state_dict.items()
    }

    nc = ckpt.get("num_classes", 4) if isinstance(ckpt, dict) else 4
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"]) if isinstance(ckpt, dict) else ["AMD", "DME",
                                                                                                      "DRUSEN",
                                                                                                      "NORMAL"]

    cls_hidden = _detect_cls_head(remapped)
    print(f"  cls_head detected: hidden={cls_hidden} ({'probed' if cls_hidden == 512 else 'original'})")

    # Ințializăm modelul folosind clasa importată din src.model.medsiglip
    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc, cls_hidden=cls_hidden)
    model.load_state_dict(remapped, strict=False)  # strict=False este util când folosim LoRA
    model = model.to(cfg.device)
    model.eval()
    print(f"  Model: {cfg.ckpt_path}")

    dataset = OCT5kDataset(
        split_csv=f"{cfg.splits_dir}/test.csv", split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=proc, mode="eval",
    )
    loader = DataLoader(
        dataset, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )
    print(f"  Dataset: {len(dataset)} imagini | Classes: {classes}\n")

    data = extract_all(model, loader, dataset)
    print(f"  Extracted {len(data['img_emb'])} embeddings\n")

    # ---- [1] R@K per disease ----
    ret_results, i2t_top, t2i_top = compute_retrieval_per_disease(
        data["img_emb"], data["txt_emb"], data["labels"], data["diseases"], classes)

    print(f"\n  {'':>15} {'Avg R@1':>8} {'Avg R@2':>8} {'Avg R@3':>8} {'I2T R@1':>8} {'T2I R@1':>8} {'Count':>6}")
    print(f"  {'-'*62}")
    for cls_name in classes:
        r = ret_results["per_class"].get(cls_name, {})
        print(f"  {cls_name:>15} {r.get('Avg_R@1', 0):>7.1f}% {r.get('Avg_R@2', 0):>7.1f}% "
              f"{r.get('Avg_R@3', 0):>7.1f}% {r.get('I2T_R@1', 0):>7.1f}% "
              f"{r.get('T2I_R@1', 0):>7.1f}% {r.get('count', 0):>6}")
    o = ret_results["overall"]
    print(f"  {'OVERALL':>15} {o['Avg_R@1']:>7.1f}% {o['Avg_R@2']:>7.1f}% "
          f"{o['Avg_R@3']:>7.1f}% {o['I2T_R@1']:>7.1f}% {o['T2I_R@1']:>7.1f}% {len(data['labels']):>6}")

    # ---- [2] Confusion matrix ----
    cm = compute_retrieval_confusion(i2t_top, data["labels"], classes)
    print(f"\n  Retrieval Confusion Matrix (I2T top-1):")
    print(f"  {'True/Retr':<12}", end="")
    for cls in classes: print(f" {cls:>8}", end="")
    print()
    for i, cls in enumerate(classes):
        print(f"  {cls:<12}", end="")
        for j in range(len(classes)):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            print(f" {pct:>7.1f}%", end="")
        print()

    # ---- [3] Failure analysis ----
    failures, per_class_stats = analyze_failures(i2t_top, data["labels"], data["diseases"], data["paths"], classes)
    n_total = len(data["labels"])
    n_r1_miss = len(failures["r1_miss_r2_hit"]) + len(failures["r1_miss_r3_hit"]) + len(failures["r1_r2_r3_miss"])
    n_r2_recovery = len(failures["r1_miss_r2_hit"])
    n_r3_recovery = len(failures["r1_miss_r3_hit"])
    n_total_miss  = len(failures["r1_r2_r3_miss"])

    print(f"\n  Failure Breakdown ({n_total} imagini):")
    print(f"    R@1 correct:           {n_total - n_r1_miss:>5} ({(n_total - n_r1_miss) / n_total * 100:.1f}%)")
    print(f"    R@1 miss, R@2 hit:     {n_r2_recovery:>5} ({n_r2_recovery / n_total * 100:.1f}%)")
    print(f"    R@1+R@2 miss, R@3 hit: {n_r3_recovery:>5} ({n_r3_recovery / n_total * 100:.1f}%)")
    print(f"    R@3 miss (total fail): {n_total_miss:>5} ({n_total_miss / n_total * 100:.1f}%)")

    print(f"\n  Per class:")
    print(f"  {'Class':<12} {'Total':>6} {'R@1':>8} {'R@2':>8} {'R@3':>8}")
    print(f"  {'-'*42}")
    for cls in classes:
        s = per_class_stats[cls]
        t = s["total"]
        if t == 0: continue
        print(f"  {cls:<12} {t:>6} {s['r1_hit']/t*100:>7.1f}% {s['r2_hit']/t*100:>7.1f}% {s['r3_hit']/t*100:>7.1f}%")

    confusion_pairs = defaultdict(int)
    for category in failures.values():
        for f in category:
            confusion_pairs[f"{f['true_class']} -> {f['retrieved_class']}"] += 1
    print(f"\n  Top confusion pairs:")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair:<25} {count} times")

    # ---- plots ----
    print(f"\n  Generating plots...")
    plot_retrieval_per_class(ret_results, classes)
    plot_confusion_matrix(cm, classes)
    plot_failure_breakdown(per_class_stats, classes)

    # ---- save ----
    output = {
        "n_images": n_total,
        "retrieval_metrics": ret_results,
        "confusion_matrix": cm.tolist(),
        "failure_counts": {
            "r1_correct": n_total - n_r1_miss,
            "r1_miss_r2_hit": n_r2_recovery,
            "r1_r2_miss_r3_hit": n_r3_recovery,
            "r3_miss": n_total_miss,
        },
        "per_class_stats": per_class_stats,
        "top_confusion_pairs": dict(sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]),
    }
    os.makedirs(os.path.dirname(cfg.out_json), exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({n_total} imagini)")
    print(f"{'=' * 70}")
    print(f"  Avg R@1={o['Avg_R@1']}% | R@2={o['Avg_R@2']}% | R@3={o['Avg_R@3']}%")
    print(f"  Results: {cfg.out_json}")
    print(f"  Figures: {cfg.fig_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()