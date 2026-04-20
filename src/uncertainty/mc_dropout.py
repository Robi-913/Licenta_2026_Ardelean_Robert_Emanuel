"""
Uncertainty Estimation — Monte Carlo Dropout pentru MedSigLIP

Ruleaza modelul de N ori pe aceeasi imagine cu dropout ACTIV.
Daca predictiile variaza mult → modelul nu e sigur.
Daca sunt consistente → modelul e confident.

Output:
    experiments/figures/uncertainty_distribution.png   — distributie confidence
    experiments/figures/uncertainty_per_class.png      — uncertainty per boala
    experiments/figures/uncertainty_examples.png       — exemple sigure vs incerte
    experiments/uncertainty_results.json               — toate scorurile

Metrici:
    - Mean confidence: cat de sigur e modelul in medie
    - Predictive entropy: cat de "imprastiate" sunt predictiile
    - Mutual information: cat de mult conteaza dropout-ul
    - Accuracy vs Uncertainty: modelul greseste mai mult cand e incert?

Rulare:
    python -m src.uncertainty.mc_dropout
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
from src.datasets.oct5k_medsiglip import OCT5kMedSigLIP, collate_medsiglip


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    model_path = "models/medsiglip-448"
    checkpoint = "experiments/medsiglip_pipeline/ckpts/best.pth"

    test_csv = "data/oct5k/splits/test.csv"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    severity_json = "data/oct5k/severity_scores.json"

    output_dir = "experiments/figures"
    results_json = "experiments/uncertainty_results.json"

    mc_samples = 20       # cate forward passes cu dropout
    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

class MedSigLIPMultiTask(nn.Module):
    def __init__(self, model_path, num_classes=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)
        emb_dim = self.model.config.vision_config.hidden_size

        self.severity_head = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def encode_image(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)


def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def enable_dropout(model):
    """Activeaza dropout-ul chiar si in eval mode — cheia MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ═══════════════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def mc_dropout_predict(model, pixel_values, n_samples):
    """
    Ruleaza N forward passes cu dropout activ.
    Returneaza: cls_probs [N, num_classes], severity_preds [N]
    """
    model.eval()
    enable_dropout(model)  # dropout ramane activ!

    all_probs = []
    all_sevs = []

    for _ in range(n_samples):
        with torch.no_grad(), autocast(cfg.device, enabled=cfg.amp):
            img_emb = model.encode_image(pixel_values)
            cls_logits = model.cls_head(img_emb)
            sev_pred = model.severity_head(img_emb).squeeze(-1)

        probs = torch.softmax(cls_logits, dim=1)
        all_probs.append(probs.cpu())
        all_sevs.append(sev_pred.cpu())

    # stack: [n_samples, batch, num_classes]
    all_probs = torch.stack(all_probs)
    all_sevs = torch.stack(all_sevs)

    return all_probs, all_sevs


def compute_uncertainty(mc_probs, mc_sevs):
    """
    Calculeaza metrici de uncertainty din MC samples.

    mc_probs: [n_samples, batch, num_classes]
    mc_sevs: [n_samples, batch]
    """
    # mean prediction
    mean_probs = mc_probs.mean(dim=0)  # [batch, num_classes]
    mean_sev = mc_sevs.mean(dim=0)     # [batch]

    # predicted class & confidence
    pred_class = mean_probs.argmax(dim=1)  # [batch]
    confidence = mean_probs.max(dim=1).values  # [batch]

    # predictive entropy: H[E[p]] — cat de "flat" e distributia medie
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)  # [batch]

    # varianta severity intre MC samples
    sev_std = mc_sevs.std(dim=0)  # [batch]

    # varianta clasificare: std al probabilitatii clasei prezise
    pred_prob_samples = torch.gather(
        mc_probs, 2, pred_class.unsqueeze(0).unsqueeze(2).expand(mc_probs.shape[0], -1, 1)
    ).squeeze(2)  # [n_samples, batch]
    cls_std = pred_prob_samples.std(dim=0)  # [batch]

    return {
        "pred_class": pred_class.numpy(),
        "confidence": confidence.numpy(),
        "entropy": entropy.numpy(),
        "cls_std": cls_std.numpy(),
        "sev_mean": mean_sev.numpy() * 100,
        "sev_std": sev_std.numpy() * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    print(f"{'=' * 70}")
    print(f"  UNCERTAINTY ESTIMATION — Monte Carlo Dropout")
    print(f"  MC samples: {cfg.mc_samples} forward passes per image")
    print(f"{'=' * 70}")

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # model
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, num_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)

    # dataset
    test_ds = OCT5kMedSigLIP(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=processor,
        mode="eval",
    )

    loader = DataLoader(test_ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate_medsiglip)

    # MC Dropout pe tot test set-ul
    all_results = {
        "pred_class": [], "true_class": [], "confidence": [],
        "entropy": [], "cls_std": [], "sev_mean": [], "sev_std": [],
        "sev_true": [],
    }

    for batch in tqdm(loader, desc="MC Dropout inference"):
        pv = batch["pixel_values"].to(cfg.device)
        labels = batch["label"].numpy()
        sev_true = batch["severity"].numpy() * 100

        mc_probs, mc_sevs = mc_dropout_predict(model, pv, cfg.mc_samples)
        unc = compute_uncertainty(mc_probs, mc_sevs)

        all_results["pred_class"].extend(unc["pred_class"])
        all_results["true_class"].extend(labels)
        all_results["confidence"].extend(unc["confidence"])
        all_results["entropy"].extend(unc["entropy"])
        all_results["cls_std"].extend(unc["cls_std"])
        all_results["sev_mean"].extend(unc["sev_mean"])
        all_results["sev_std"].extend(unc["sev_std"])
        all_results["sev_true"].extend(sev_true)

        del pv, mc_probs, mc_sevs

    free_memory()

    # convert to numpy
    for k in all_results:
        all_results[k] = np.array(all_results[k])

    pred = all_results["pred_class"]
    true = all_results["true_class"]
    conf = all_results["confidence"]
    entropy = all_results["entropy"]
    cls_std = all_results["cls_std"]
    sev_std = all_results["sev_std"]

    correct = (pred == true)
    accuracy = correct.mean() * 100

    # ═══════════════════════════════════════════════════════════════
    # STATISTICI
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─' * 50}")
    print(f"  RESULTS:")
    print(f"{'─' * 50}")
    print(f"  Total images: {len(pred)}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Mean confidence: {conf.mean()*100:.1f}%")
    print(f"  Mean entropy: {entropy.mean():.3f}")
    print(f"  Mean cls_std: {cls_std.mean():.4f}")
    print(f"  Mean sev_std: {sev_std.mean():.1f}%")

    # correct vs incorrect
    print(f"\n  Correct predictions:")
    print(f"    Confidence: {conf[correct].mean()*100:.1f}% ± {conf[correct].std()*100:.1f}%")
    print(f"    Entropy: {entropy[correct].mean():.3f}")
    print(f"    Severity std: {sev_std[correct].mean():.1f}%")

    print(f"\n  Incorrect predictions:")
    if (~correct).sum() > 0:
        print(f"    Confidence: {conf[~correct].mean()*100:.1f}% ± {conf[~correct].std()*100:.1f}%")
        print(f"    Entropy: {entropy[~correct].mean():.3f}")
        print(f"    Severity std: {sev_std[~correct].mean():.1f}%")
    else:
        print(f"    No errors!")

    # per class
    print(f"\n  Per class uncertainty:")
    for i, cls_name in enumerate(classes):
        mask = true == i
        if mask.sum() > 0:
            print(f"    {cls_name}: conf={conf[mask].mean()*100:.1f}% | "
                  f"entropy={entropy[mask].mean():.3f} | "
                  f"sev_std={sev_std[mask].mean():.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 1: Confidence Distribution (correct vs incorrect)
    # ═══════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(conf[correct] * 100, bins=30, alpha=0.7, color="green", label="Correct", density=True)
    if (~correct).sum() > 0:
        axes[0, 0].hist(conf[~correct] * 100, bins=30, alpha=0.7, color="red", label="Incorrect", density=True)
    axes[0, 0].set_title("Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence %")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # PLOT 2: Entropy Distribution
    axes[0, 1].hist(entropy[correct], bins=30, alpha=0.7, color="green", label="Correct", density=True)
    if (~correct).sum() > 0:
        axes[0, 1].hist(entropy[~correct], bins=30, alpha=0.7, color="red", label="Incorrect", density=True)
    axes[0, 1].set_title("Predictive Entropy")
    axes[0, 1].set_xlabel("Entropy")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # PLOT 3: Uncertainty per Class
    class_confs = [conf[true == i] * 100 for i in range(len(classes))]
    bp = axes[1, 0].boxplot(class_confs, labels=classes, patch_artist=True)
    colors_box = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 0].set_title("Confidence per Disease Class")
    axes[1, 0].set_ylabel("Confidence %")
    axes[1, 0].grid(alpha=0.3)

    # PLOT 4: Severity Uncertainty
    axes[1, 1].scatter(all_results["sev_mean"], sev_std, alpha=0.4, s=15, c=conf, cmap="RdYlGn")
    axes[1, 1].set_title("Severity: Mean vs Uncertainty")
    axes[1, 1].set_xlabel("Predicted Severity %")
    axes[1, 1].set_ylabel("Severity Std (uncertainty)")
    axes[1, 1].grid(alpha=0.3)
    cb = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cb.set_label("Classification Confidence")

    plt.suptitle(f"MedSigLIP Uncertainty — MC Dropout ({cfg.mc_samples} samples)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/uncertainty_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot: {cfg.output_dir}/uncertainty_analysis.png")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 5: Reliability Diagram (calibration)
    # ═══════════════════════════════════════════════════════════════

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(conf[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)

    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#3498db", label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(
        (count / total) * abs(acc - conf_val)
        for count, acc, conf_val in zip(bin_counts, bin_accs, bin_confs)
        if count > 0
    )
    ax.text(0.05, 0.9, f"ECE = {ece:.4f}", fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/calibration_diagram.png", dpi=200)
    plt.close()
    print(f"  Plot: {cfg.output_dir}/calibration_diagram.png")

    # ═══════════════════════════════════════════════════════════════
    # SAVE JSON
    # ═══════════════════════════════════════════════════════════════

    summary = {
        "mc_samples": cfg.mc_samples,
        "total_images": len(pred),
        "accuracy": round(accuracy, 2),
        "mean_confidence": round(float(conf.mean() * 100), 2),
        "mean_entropy": round(float(entropy.mean()), 4),
        "mean_cls_std": round(float(cls_std.mean()), 4),
        "mean_sev_std": round(float(sev_std.mean()), 2),
        "ece": round(float(ece), 4),
        "correct_confidence": round(float(conf[correct].mean() * 100), 2),
        "incorrect_confidence": round(float(conf[~correct].mean() * 100), 2) if (~correct).sum() > 0 else None,
        "per_class": {},
    }

    for i, cls_name in enumerate(classes):
        mask = true == i
        if mask.sum() > 0:
            summary["per_class"][cls_name] = {
                "count": int(mask.sum()),
                "accuracy": round(float(correct[mask].mean() * 100), 2),
                "mean_confidence": round(float(conf[mask].mean() * 100), 2),
                "mean_entropy": round(float(entropy[mask].mean()), 4),
                "mean_sev_std": round(float(sev_std[mask].mean()), 2),
            }

    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Results: {cfg.results_json}")

    print(f"\n{'=' * 70}")
    print(f"  UNCERTAINTY ESTIMATION COMPLETE")
    print(f"  Accuracy: {accuracy:.1f}% | Confidence: {conf.mean()*100:.1f}% | ECE: {ece:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()