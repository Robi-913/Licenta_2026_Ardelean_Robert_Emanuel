"""
Metrici:
    - Mean confidence: cat de sigur e modelul in medie
    - Predictive entropy: cat de "imprastiate" sunt predictiile
    - Mutual information: cat de mult conteaza dropout-ul
    - Accuracy vs Uncertainty: modelul greseste mai mult cand e incert?
"""

import os
import sys
import json
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.seed import set_seed
from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k


# ---------- config ----------

class Config:
    model_path = "models/medsiglip-448"
    ckpt_path = "experiments/medsiglip_v3/ckpts/best.pth"

    test_csv = "data/oct5k/splits/test.csv"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    sev_json = "data/oct5k/severity_scores.json"

    fig_dir = "experiments/figures/uncertainty"
    out_json = "experiments/uncertainty_results.json"

    mc_passes = 20
    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.fig_dir, exist_ok=True)


# ---------- model ----------

class MedSigLIPMultiTask(nn.Module):

    def __init__(self, model_path, n_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        dim = self.backbone.config.vision_config.hidden_size

        self.sev_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)


def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def turn_on_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------- mc dropout ----------

def mc_predict(model, pv, n):
    model.eval()
    turn_on_dropout(model)

    probs_all = []
    sevs_all = []

    for _ in range(n):
        with torch.no_grad(), autocast(cfg.device, enabled=cfg.amp):
            emb = model.encode_image(pv)
            logits = model.cls_head(emb)
            sp = model.sev_head(emb).squeeze(-1).clamp(0, 1)

        probs_all.append(torch.softmax(logits, dim=1).cpu())
        sevs_all.append(sp.cpu())

    return torch.stack(probs_all), torch.stack(sevs_all)


def calc_uncertainty(mc_probs, mc_sevs):
    mean_p = mc_probs.mean(dim=0)
    mean_s = mc_sevs.mean(dim=0)

    pred = mean_p.argmax(dim=1)
    conf = mean_p.max(dim=1).values

    ent = -(mean_p * torch.log(mean_p + 1e-10)).sum(dim=1)

    s_std = mc_sevs.std(dim=0)

    pred_probs = torch.gather(
        mc_probs, 2,
        pred.unsqueeze(0).unsqueeze(2).expand(mc_probs.shape[0], -1, 1),
    ).squeeze(2)
    c_std = pred_probs.std(dim=0)

    return {
        "pred_class": pred.numpy(),
        "confidence": conf.numpy(),
        "entropy": ent.numpy(),
        "cls_std": c_std.numpy(),
        "sev_mean": mean_s.numpy() * 100,
        "sev_std": s_std.numpy() * 100,
    }


# ---------- main ----------

def main():
    set_seed()

    print(f"{'=' * 70}")
    print(f"  UNCERTAINTY ESTIMATION - Monte Carlo Dropout")
    print(f"  MC samples: {cfg.mc_passes} forward passes per image")
    print(f"{'=' * 70}")

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(cfg.device)

    ds = OCT5kDataset(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.sev_json,
        processor=proc,
        mode="eval",
    )

    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate_oct5k)

    collected = {
        "pred_class": [], "true_class": [], "confidence": [],
        "entropy": [], "cls_std": [], "sev_mean": [], "sev_std": [],
        "sev_true": [],
    }

    for batch in tqdm(loader, desc="MC Dropout"):
        pv = batch["pixel_values"].to(cfg.device)
        lbls = batch["label"].numpy()
        st = batch["severity"].numpy() * 100

        mc_p, mc_s = mc_predict(model, pv, cfg.mc_passes)
        unc = calc_uncertainty(mc_p, mc_s)

        collected["pred_class"].extend(unc["pred_class"])
        collected["true_class"].extend(lbls)
        collected["confidence"].extend(unc["confidence"])
        collected["entropy"].extend(unc["entropy"])
        collected["cls_std"].extend(unc["cls_std"])
        collected["sev_mean"].extend(unc["sev_mean"])
        collected["sev_std"].extend(unc["sev_std"])
        collected["sev_true"].extend(st)

        del pv, mc_p, mc_s

    clear_mem()

    for k in collected:
        collected[k] = np.array(collected[k])

    pred = collected["pred_class"]
    true = collected["true_class"]
    conf = collected["confidence"]
    ent = collected["entropy"]
    c_std = collected["cls_std"]
    s_std = collected["sev_std"]

    hit = pred == true
    acc = hit.mean() * 100

    # ---------- stats ----------

    print(f"\n{'─' * 50}")
    print(f"  RESULTS:")
    print(f"{'─' * 50}")
    print(f"  Total images: {len(pred)}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Mean confidence: {conf.mean() * 100:.1f}%")
    print(f"  Mean entropy: {ent.mean():.3f}")
    print(f"  Mean cls_std: {c_std.mean():.4f}")
    print(f"  Mean sev_std: {s_std.mean():.1f}%")

    print(f"\n  Correct predictions:")
    print(f"    Confidence: {conf[hit].mean() * 100:.1f}% +/- {conf[hit].std() * 100:.1f}%")
    print(f"    Entropy: {ent[hit].mean():.3f}")
    print(f"    Severity std: {s_std[hit].mean():.1f}%")

    print(f"\n  Incorrect predictions:")
    if (~hit).sum() > 0:
        print(f"    Confidence: {conf[~hit].mean() * 100:.1f}% +/- {conf[~hit].std() * 100:.1f}%")
        print(f"    Entropy: {ent[~hit].mean():.3f}")
        print(f"    Severity std: {s_std[~hit].mean():.1f}%")
    else:
        print(f"    No errors!")

    print(f"\n  Per class:")
    for i, name in enumerate(classes):
        mask = true == i
        if mask.sum() > 0:
            print(
                f"    {name}: conf={conf[mask].mean() * 100:.1f}% | "
                f"entropy={ent[mask].mean():.3f} | "
                f"sev_std={s_std[mask].mean():.1f}%"
            )

    # ---------- plot 1: confidence + entropy ----------

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(conf[hit] * 100, bins=30, alpha=0.7, color="green", label="Correct", density=True)
    if (~hit).sum() > 0:
        axes[0, 0].hist(conf[~hit] * 100, bins=30, alpha=0.7, color="red", label="Incorrect", density=True)
    axes[0, 0].set_title("Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence %")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(ent[hit], bins=30, alpha=0.7, color="green", label="Correct", density=True)
    if (~hit).sum() > 0:
        axes[0, 1].hist(ent[~hit], bins=30, alpha=0.7, color="red", label="Incorrect", density=True)
    axes[0, 1].set_title("Predictive Entropy")
    axes[0, 1].set_xlabel("Entropy")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    box_data = [conf[true == i] * 100 for i in range(len(classes))]
    bp = axes[1, 0].boxplot(box_data, labels=classes, patch_artist=True)
    box_colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    axes[1, 0].set_title("Confidence per Disease Class")
    axes[1, 0].set_ylabel("Confidence %")
    axes[1, 0].grid(alpha=0.3)

    sc = axes[1, 1].scatter(
        collected["sev_mean"], s_std,
        alpha=0.4, s=15, c=conf, cmap="RdYlGn",
    )
    axes[1, 1].set_title("Severity: Mean vs Uncertainty")
    axes[1, 1].set_xlabel("Predicted Severity %")
    axes[1, 1].set_ylabel("Severity Std (uncertainty)")
    axes[1, 1].grid(alpha=0.3)
    cb = plt.colorbar(sc, ax=axes[1, 1])
    cb.set_label("Classification Confidence")

    plt.suptitle(
        f"MedSigLIP Uncertainty - MC Dropout ({cfg.mc_passes} samples)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/uncertainty_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot: {cfg.fig_dir}/uncertainty_analysis.png")

    # ---------- plot 2: calibration ----------

    fig, ax = plt.subplots(figsize=(8, 8))

    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_n = [], [], []

    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if mask.sum() > 0:
            bin_acc.append(hit[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_n.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_conf.append((edges[i] + edges[i + 1]) / 2)
            bin_n.append(0)

    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)

    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#3498db", label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    total_n = sum(bin_n)
    ece = sum(
        (cnt / total_n) * abs(a - c)
        for cnt, a, c in zip(bin_n, bin_acc, bin_conf)
        if cnt > 0
    )
    ax.text(
        0.05, 0.9, f"ECE = {ece:.4f}", fontsize=14,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/calibration_diagram.png", dpi=200)
    plt.close()
    print(f"  Plot: {cfg.fig_dir}/calibration_diagram.png")

    # ---------- save ----------

    summary = {
        "mc_samples": cfg.mc_passes,
        "total_images": len(pred),
        "accuracy": round(acc, 2),
        "mean_confidence": round(float(conf.mean() * 100), 2),
        "mean_entropy": round(float(ent.mean()), 4),
        "mean_cls_std": round(float(c_std.mean()), 4),
        "mean_sev_std": round(float(s_std.mean()), 2),
        "ece": round(float(ece), 4),
        "correct_confidence": round(float(conf[hit].mean() * 100), 2),
        "incorrect_confidence": round(float(conf[~hit].mean() * 100), 2) if (~hit).sum() > 0 else None,
        "per_class": {},
    }

    for i, name in enumerate(classes):
        mask = true == i
        if mask.sum() > 0:
            summary["per_class"][name] = {
                "count": int(mask.sum()),
                "accuracy": round(float(hit[mask].mean() * 100), 2),
                "mean_confidence": round(float(conf[mask].mean() * 100), 2),
                "mean_entropy": round(float(ent[mask].mean()), 4),
                "mean_sev_std": round(float(s_std[mask].mean()), 2),
            }

    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Results: {cfg.out_json}")

    print(f"\n{'=' * 70}")
    print(f"  UNCERTAINTY ESTIMATION COMPLETE")
    print(f"  Accuracy: {acc:.1f}% | Confidence: {conf.mean() * 100:.1f}% | ECE: {ece:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()