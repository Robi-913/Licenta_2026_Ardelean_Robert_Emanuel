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
from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k


# ---------- config ----------

class Config:
    model_path = "models/medsiglip-448"
    ckpt_path = "experiments/medsiglip_v3/ckpts/best.pth"

    test_csv = "data/oct5k/splits/test.csv"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    sev_json = "data/oct5k/severity_scores.json"

    fig_dir = "experiments/figures/tsne"

    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()

    perplexity = 30
    tsne_iter = 1000


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


# ---------- extract embeddings ----------

@torch.no_grad()
def get_embeddings(model, loader):
    model.eval()
    embs, lbls, sevs = [], [], []
    sp_all, cp_all = [], []

    for batch in tqdm(loader, desc="  Extracting"):
        pv = batch["pixel_values"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie = model.encode_image(pv)
            sp = model.sev_head(ie).squeeze(-1).clamp(0, 1)
            cp = model.cls_head(ie).argmax(1)

        embs.append(ie.cpu().numpy())
        lbls.append(batch["label"].numpy())
        sevs.append(batch["severity"].numpy() * 100)
        sp_all.append(sp.cpu().numpy() * 100)
        cp_all.append(cp.cpu().numpy())

        del pv, ie, sp, cp

    clear_mem()

    return {
        "emb": np.concatenate(embs),
        "labels": np.concatenate(lbls),
        "sev_true": np.concatenate(sevs),
        "sev_pred": np.concatenate(sp_all),
        "cls_pred": np.concatenate(cp_all),
    }


# ---------- plots ----------

CLS_COLORS = {"AMD": "#e74c3c", "DME": "#3498db", "DRUSEN": "#f39c12", "NORMAL": "#2ecc71"}
CLS_MARKERS = {"AMD": "o", "DME": "s", "DRUSEN": "^", "NORMAL": "D"}


def plot_by_disease(pts, labels, classes):
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(classes):
        mask = labels == i
        ax.scatter(
            pts[mask, 0], pts[mask, 1],
            c=CLS_COLORS.get(name, "#999"),
            marker=CLS_MARKERS.get(name, "o"),
            label=f"{name} ({mask.sum()})",
            alpha=0.6, s=30, edgecolors="white", linewidth=0.3,
        )

    ax.set_title("MedSigLIP Embeddings - t-SNE by Disease", fontsize=14)
    ax.legend(fontsize=12, markerscale=1.5)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/tsne_by_disease.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/tsne_by_disease.png")


def plot_by_severity(pts, sev, labels, classes):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc = axes[0].scatter(
        pts[:, 0], pts[:, 1],
        c=sev, cmap="RdYlGn_r", alpha=0.6, s=30,
        edgecolors="white", linewidth=0.3,
        vmin=0, vmax=100,
    )
    plt.colorbar(sc, ax=axes[0], label="Severity %")
    axes[0].set_title("t-SNE by Severity (all)", fontsize=13)
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    axes[0].grid(alpha=0.2)

    for i, name in enumerate(classes):
        mask = labels == i
        axes[1].scatter(
            pts[mask, 0], pts[mask, 1],
            c=sev[mask], cmap="RdYlGn_r", alpha=0.6, s=30,
            edgecolors=CLS_COLORS.get(name, "#999"), linewidth=0.8,
            vmin=0, vmax=100, label=name,
        )

    axes[1].set_title("t-SNE by Severity (per class borders)", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/tsne_by_severity.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/tsne_by_severity.png")


def plot_predictions(pts, labels, preds, classes):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    palette = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

    for ax, data, title in [(axes[0], labels, "True Labels"), (axes[1], preds, "Predicted Labels")]:
        for i, name in enumerate(classes):
            mask = data == i
            ax.scatter(
                pts[mask, 0], pts[mask, 1],
                c=palette[i], label=name,
                alpha=0.6, s=30, edgecolors="white", linewidth=0.3,
            )
        ax.set_title(f"MedSigLIP - {title}", fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(alpha=0.2)

    wrong = labels != preds
    if wrong.sum() > 0:
        axes[1].scatter(
            pts[wrong, 0], pts[wrong, 1],
            facecolors="none", edgecolors="black", s=100,
            linewidth=2, label=f"Errors ({wrong.sum()})", zorder=5,
        )
        axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/tsne_predictions.png", dpi=200)
    plt.close()
    print(f"  Saved: {cfg.fig_dir}/tsne_predictions.png")


# ---------- main ----------

def main():
    set_seed()

    print(f"{'=' * 70}")
    print("  t-SNE VISUALIZATION: MedSigLIP Embeddings")
    print(f"{'=' * 70}")

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(cfg.device)
    model.eval()

    ds = OCT5kDataset(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.sev_json,
        processor=proc,
        mode="eval",
    )

    loader = DataLoader(
        ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_oct5k,
    )

    data = get_embeddings(model, loader)
    print(f"  Got {len(data['emb'])} embeddings, dim={data['emb'].shape[1]}")

    print(f"  Running t-SNE (perplexity={cfg.perplexity})...")
    reducer = TSNE(
        n_components=2,
        perplexity=cfg.perplexity,
        max_iter=cfg.tsne_iter,
        random_state=42,
        init="pca",
    )
    pts = reducer.fit_transform(data["emb"])
    print("  t-SNE done!")

    print("\n  Generating plots...")
    plot_by_disease(pts, data["labels"], classes)
    plot_by_severity(pts, data["sev_true"], data["labels"], classes)
    plot_predictions(pts, data["labels"], data["cls_pred"], classes)

    print("\n  Cluster spread:")
    for i, name in enumerate(classes):
        mask = data["labels"] == i
        if mask.sum() < 2:
            continue
        cluster = pts[mask]
        center = cluster.mean(axis=0)
        spread = np.sqrt(((cluster - center) ** 2).sum(axis=1).mean())
        print(f"    {name}: {mask.sum()} pts, spread={spread:.1f}")

    print(f"\n{'=' * 70}")
    print(f"  Plots saved to: {cfg.fig_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()