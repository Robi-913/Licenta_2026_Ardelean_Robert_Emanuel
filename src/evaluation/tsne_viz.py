"""
t-SNE / UMAP Vizualizare Embeddings MedSigLIP

Genereaza scatter plots cu embedding-urile imaginilor, colorate per:
  1. Clasa de boala (AMD, DME, DRUSEN, NORMAL)
  2. Severity (gradient de culoare 0-100%)

Arata cat de bine separa modelul clasele in spatiul latent.

Output:
    experiments/figures/tsne_by_disease.png
    experiments/figures/tsne_by_severity.png

Rulare:
    python -m src.evaluation.tsne_viz
"""

import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
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

    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()

    # t-SNE params
    tsne_perplexity = 30
    tsne_n_iter = 1000


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


# ═══════════════════════════════════════════════════════════════════════
# EXTRACT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    all_emb, all_lbl, all_sev = [], [], []
    all_sev_pred, all_cls_pred = [], []

    for batch in tqdm(loader, desc="  Extracting embeddings"):
        pv = batch["pixel_values"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            img_emb = model.encode_image(pv)
            sev_pred = model.severity_head(img_emb).squeeze(-1)
            cls_pred = model.cls_head(img_emb).argmax(1)

        all_emb.append(img_emb.cpu().numpy())
        all_lbl.append(batch["label"].numpy())
        all_sev.append(batch["severity"].numpy() * 100)
        all_sev_pred.append(sev_pred.cpu().numpy() * 100)
        all_cls_pred.append(cls_pred.cpu().numpy())

        del pv, img_emb, sev_pred, cls_pred

    free_memory()

    return {
        "embeddings": np.concatenate(all_emb),
        "labels": np.concatenate(all_lbl),
        "severity_true": np.concatenate(all_sev),
        "severity_pred": np.concatenate(all_sev_pred),
        "cls_pred": np.concatenate(all_cls_pred),
    }


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_tsne_by_disease(tsne_2d, labels, classes):
    """t-SNE colorat per clasa de boala."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = {"AMD": "#e74c3c", "DME": "#3498db", "DRUSEN": "#f39c12", "NORMAL": "#2ecc71"}
    markers = {"AMD": "o", "DME": "s", "DRUSEN": "^", "NORMAL": "D"}

    for i, cls_name in enumerate(classes):
        mask = labels == i
        ax.scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=colors.get(cls_name, "#999"),
            marker=markers.get(cls_name, "o"),
            label=f"{cls_name} ({mask.sum()})",
            alpha=0.6, s=30, edgecolors="white", linewidth=0.3,
        )

    ax.set_title("MedSigLIP Embeddings — t-SNE by Disease Class", fontsize=14)
    ax.legend(fontsize=12, markerscale=1.5)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/tsne_by_disease.png", dpi=200)
    plt.close()
    print(f"  Salvat: {cfg.output_dir}/tsne_by_disease.png")


def plot_tsne_by_severity(tsne_2d, severity, labels, classes):
    """t-SNE colorat per severity (gradient)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: severity colormap pe toate
    sc = axes[0].scatter(
        tsne_2d[:, 0], tsne_2d[:, 1],
        c=severity, cmap="RdYlGn_r", alpha=0.6, s=30,
        edgecolors="white", linewidth=0.3,
        vmin=0, vmax=100,
    )
    plt.colorbar(sc, ax=axes[0], label="Severity %")
    axes[0].set_title("t-SNE by Severity (all classes)", fontsize=13)
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    axes[0].grid(alpha=0.2)

    # Plot 2: severity per clasa (subplots)
    colors_cls = {"AMD": "#e74c3c", "DME": "#3498db", "DRUSEN": "#f39c12", "NORMAL": "#2ecc71"}
    for i, cls_name in enumerate(classes):
        mask = labels == i
        axes[1].scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=severity[mask], cmap="RdYlGn_r", alpha=0.6, s=30,
            edgecolors=colors_cls.get(cls_name, "#999"), linewidth=0.8,
            vmin=0, vmax=100, label=cls_name,
        )

    axes[1].set_title("t-SNE by Severity (per class borders)", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/tsne_by_severity.png", dpi=200)
    plt.close()
    print(f"  Salvat: {cfg.output_dir}/tsne_by_severity.png")


def plot_tsne_predictions(tsne_2d, labels, cls_pred, classes):
    """t-SNE: true labels vs predicted labels side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

    for ax, data, title in [
        (axes[0], labels, "True Labels"),
        (axes[1], cls_pred, "Predicted Labels"),
    ]:
        for i, cls_name in enumerate(classes):
            mask = data == i
            ax.scatter(
                tsne_2d[mask, 0], tsne_2d[mask, 1],
                c=colors[i], label=cls_name,
                alpha=0.6, s=30, edgecolors="white", linewidth=0.3,
            )
        ax.set_title(f"MedSigLIP — {title}", fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(alpha=0.2)

    # highlight errors
    errors = labels != cls_pred
    if errors.sum() > 0:
        axes[1].scatter(
            tsne_2d[errors, 0], tsne_2d[errors, 1],
            facecolors="none", edgecolors="black", s=100,
            linewidth=2, label=f"Errors ({errors.sum()})", zorder=5,
        )
        axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/tsne_predictions.png", dpi=200)
    plt.close()
    print(f"  Salvat: {cfg.output_dir}/tsne_predictions.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    print(f"{'=' * 70}")
    print("  t-SNE VISUALIZATION: MedSigLIP Embeddings")
    print(f"{'=' * 70}")

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # model
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, num_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)
    model.eval()

    # dataset
    test_ds = OCT5kMedSigLIP(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=processor,
        mode="eval",
    )

    loader = DataLoader(
        test_ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_medsiglip,
    )

    # extract embeddings
    data = extract_embeddings(model, loader)
    print(f"  Extracted {len(data['embeddings'])} embeddings, dim={data['embeddings'].shape[1]}")

    # t-SNE
    print(f"  Running t-SNE (perplexity={cfg.tsne_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        n_iter=cfg.tsne_n_iter,
        random_state=42,
        init="pca",
    )
    tsne_2d = tsne.fit_transform(data["embeddings"])
    print(f"  t-SNE done!")

    # plots
    print(f"\n  Generating plots...")
    plot_tsne_by_disease(tsne_2d, data["labels"], classes)
    plot_tsne_by_severity(tsne_2d, data["severity_true"], data["labels"], classes)
    plot_tsne_predictions(tsne_2d, data["labels"], data["cls_pred"], classes)

    # statistici clustering
    print(f"\n  Clustering quality:")
    for i, cls_name in enumerate(classes):
        mask = data["labels"] == i
        if mask.sum() > 1:
            cluster_points = tsne_2d[mask]
            centroid = cluster_points.mean(axis=0)
            spread = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1).mean())
            print(f"    {cls_name}: {mask.sum()} pts, spread={spread:.1f}")

    print(f"\n{'=' * 70}")
    print(f"  Plots salvate in: {cfg.output_dir}/")
    print(f"    tsne_by_disease.png     - colorate per clasa")
    print(f"    tsne_by_severity.png    - colorate per severity")
    print(f"    tsne_predictions.png    - true vs predicted + errors")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()