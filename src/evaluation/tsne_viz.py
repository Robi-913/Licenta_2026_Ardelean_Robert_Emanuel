import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


class Config:
    model_path = "models/medsiglip-448"
    ckpt_path = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"

    test_csv = "data/oct5k/splits_v3/test.csv"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    sev_json = "data/oct5k/severity_scores_v2.json"

    fig_dir = "experiments/figures/tsne_v13"

    batch_size = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()

    tsne_perplexity = 30
    tsne_max_iter = 1000


cfg = Config()
os.makedirs(cfg.fig_dir, exist_ok=True)

# Culori si markere constante per clasa — folosite in toate graficele
CLS_COLORS = {"AMD": "#e74c3c", "DME": "#3498db", "DRUSEN": "#f39c12", "NORMAL": "#2ecc71"}
CLS_MARKERS = {"AMD": "o", "DME": "s", "DRUSEN": "^", "NORMAL": "D"}



def _detect_cls_hidden(state_dict: dict) -> int:
    """
    Detecteaza daca cls_head e cel original (256) sau cel din linear_probe (512).
    Ne uitam la dimensiunea primului Linear din capul de clasificare.
    """
    for key in ["classification_head.1.weight", "cls_head.1.weight"]:
        w = state_dict.get(key)
        if w is not None:
            return w.shape[0]
    return 256


def load_model(ckpt_path: str) -> tuple[MedSigLIPMultiTask, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    n_classes = ckpt.get("num_classes", 4) if isinstance(ckpt, dict) else 4
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"]) if isinstance(ckpt, dict) else ["AMD", "DME",
                                                                                                      "DRUSEN",
                                                                                                      "NORMAL"]
    cls_hidden = _detect_cls_hidden(state_dict)

    print(f"  cls_head hidden_dim detectat: {cls_hidden}")

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=n_classes, cls_hidden=cls_hidden)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(cfg.device).eval()
    return model, classes


# EXTRAGERE EMBEDDINGS

@torch.no_grad()
def extract_embeddings(model: MedSigLIPMultiTask, loader: DataLoader) -> dict:
    """
    Ruleaza encoder vizual pe tot loader-ul si colecteaza:
    - embedding-urile L2-normalizate
    - label-urile adevarate si prezise
    - severitatea adevarata si prezisa
    """
    all_embs, all_labels, all_sev_true = [], [], []
    all_sev_pred, all_cls_pred = [], []

    for batch in tqdm(loader, desc="  Extracting embeddings"):
        pv = batch["pixel_values"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.use_amp):
            # encode_image returneaza features brute — normalizam pt embedding space
            image_pooled = model.encode_image(pv)
            sev_pred = model.severity_head(image_pooled).squeeze(-1).clamp(0, 1)
            cls_pred = model.classification_head(image_pooled).argmax(1)

        all_embs.append(image_pooled.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_sev_true.append(batch["severity"].numpy() * 100)  # scalam la [0, 100]
        all_sev_pred.append(sev_pred.cpu().numpy() * 100)
        all_cls_pred.append(cls_pred.cpu().numpy())

    _free_mem()

    return {
        "emb": np.concatenate(all_embs),
        "labels": np.concatenate(all_labels),
        "sev_true": np.concatenate(all_sev_true),
        "sev_pred": np.concatenate(all_sev_pred),
        "cls_pred": np.concatenate(all_cls_pred),
    }


# GRAFICE

def _scatter_kwargs(color, marker="o") -> dict:
    """Parametri comuni pt scatter-urile de mai jos."""
    return dict(c=color, marker=marker, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)


def plot_by_disease(pts: np.ndarray, labels: np.ndarray, classes: list) -> None:
    """Cluster-e t-SNE colorate pe clasa de boala."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(classes):
        mask = labels == i
        ax.scatter(
            pts[mask, 0], pts[mask, 1],
            label=f"{name} ({mask.sum()})",
            **_scatter_kwargs(CLS_COLORS.get(name, "#999"), CLS_MARKERS.get(name, "o")),
        )

    ax.set(title="MedSigLIP v13 — t-SNE by Disease", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
    ax.legend(fontsize=12, markerscale=1.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, "tsne_by_disease.png")


def plot_by_severity(pts: np.ndarray, sev: np.ndarray, labels: np.ndarray, classes: list) -> None:
    """Gradient de culoare pe scorul de severitate — global si cu borduri per clasa."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Global — colormap pe severitate
    sc = axes[0].scatter(pts[:, 0], pts[:, 1], c=sev, cmap="RdYlGn_r",
                         alpha=0.6, s=30, edgecolors="white", linewidth=0.3, vmin=0, vmax=100)
    plt.colorbar(sc, ax=axes[0], label="Severity %")
    axes[0].set(title="t-SNE by Severity (all)", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
    axes[0].grid(alpha=0.2)

    # Per clasa — bordura colorata cu culoarea clasei, interior = severitate
    for i, name in enumerate(classes):
        mask = labels == i
        axes[1].scatter(pts[mask, 0], pts[mask, 1], c=sev[mask], cmap="RdYlGn_r",
                        alpha=0.6, s=30, edgecolors=CLS_COLORS.get(name, "#999"),
                        linewidth=0.8, vmin=0, vmax=100, label=name)
    axes[1].set(title="t-SNE by Severity (per class borders)", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, "tsne_by_severity.png")


def plot_predictions(pts: np.ndarray, labels: np.ndarray, preds: np.ndarray, classes: list) -> None:
    """True labels vs predicted labels — erorile marcate cu cerc negru."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    palette = [CLS_COLORS.get(c, "#999") for c in classes]

    for ax, data, title in [(axes[0], labels, "True Labels"), (axes[1], preds, "Predicted Labels")]:
        for i, name in enumerate(classes):
            mask = data == i
            ax.scatter(pts[mask, 0], pts[mask, 1], c=palette[i], label=name,
                       alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
        ax.set(title=f"MedSigLIP v15 — {title}", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.2)

    # Marcam erorile de clasificare cu cerc negru deasupra
    wrong = labels != preds
    if wrong.any():
        axes[1].scatter(pts[wrong, 0], pts[wrong, 1], facecolors="none", edgecolors="black",
                        s=100, linewidth=2, label=f"Errors ({wrong.sum()})", zorder=5)
        axes[1].legend(fontsize=11)

    plt.tight_layout()
    _save(fig, "tsne_predictions.png")


def _save(fig: plt.Figure, filename: str) -> None:
    path = f"{cfg.fig_dir}/{filename}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")

def _free_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _print_cluster_stats(pts: np.ndarray, labels: np.ndarray, classes: list) -> None:
    """Afiseaza spread-ul fiecarui cluster (distanta medie fata de centru)."""
    print("\n  Cluster spread:")
    for i, name in enumerate(classes):
        mask = labels == i
        if mask.sum() < 2:
            continue
        cluster = pts[mask]
        center = cluster.mean(axis=0)
        spread = np.sqrt(((cluster - center) ** 2).sum(axis=1).mean())
        print(f"    {name}: {mask.sum()} pts, spread={spread:.1f}")

def main():
    set_seed()

    print("  t-SNE VISUALIZATION — MedSigLIP v13 Embeddings")
    print(f"  Checkpoint: {cfg.ckpt_path}")

    processor = AutoProcessor.from_pretrained(cfg.model_path)
    model, classes = load_model(cfg.ckpt_path)

    ds = OCT5kDataset(
        split_csv=cfg.test_csv, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=processor, mode="eval",
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    data = extract_embeddings(model, loader)
    print(f"  {len(data['emb'])} embeddings extrase, dim={data['emb'].shape[1]}")

    print(f"\n  t-SNE (perplexity={cfg.tsne_perplexity}, iter={cfg.tsne_max_iter})...")
    pts = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        max_iter=cfg.tsne_max_iter,
        random_state=42,
        init="pca",
    ).fit_transform(data["emb"])
    print("  t-SNE gata!")

    print("\n  Generare grafice...")
    plot_by_disease(pts, data["labels"], classes)
    plot_by_severity(pts, data["sev_true"], data["labels"], classes)
    plot_predictions(pts, data["labels"], data["cls_pred"], classes)

    _print_cluster_stats(pts, data["labels"], classes)

    print(f"\n  Plots salvate in: {cfg.fig_dir}/")


if __name__ == "__main__":
    main()
