"""
Retrieval Demo — MedSigLIP v3

Genereaza vizualizari de retrieval:
  1. I2T Grid: imagine query -> top 3 texte gasite
  2. T2I Grid: text query -> top 3 imagini gasite
  3. Per-class R@1, R@5, R@10 breakdown
  4. Similarity distribution histogram
  5. Failure analysis

Rulare:
    python src/retrieval/retrieval_demo.py
"""

import os
import sys
import json
import gc
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.seed import set_seed
from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k


# ---------- config ----------

class Config:
    model_path = "models/medsiglip-448"
    checkpoint = "experiments/medsiglip_v3/ckpts/best.pth"

    test_csv = "data/oct5k/splits/test.csv"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    severity_json = "data/oct5k/severity_scores.json"

    output_dir = "experiments/figures/retrieval"
    results_json = "experiments/retrieval_results.json"

    samples_per_class = 2
    top_k = 3

    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ---------- cross-attention fusion ----------

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
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)
        attn_a, _ = self.attn_a2b(query=a, key=b, value=b)
        attn_b, _ = self.attn_b2a(query=b, key=a, value=a)
        attn_a = attn_a.squeeze(1)
        attn_b = attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


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
        self.fusion = CrossAttentionFusion(dim, heads=4, dropout=0.1)

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)


def free_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------- extract all embeddings ----------

@torch.no_grad()
def extract_all(model, loader):
    model.eval()
    all_img, all_txt, all_lbl, all_sev = [], [], [], []

    for batch in tqdm(loader, desc="  Extracting embeddings"):
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
        all_sev.append(batch["severity"])

        del pv, ia, ma, ib, mb, ie, ea, eb, merged

    free_mem()

    return {
        "img_emb": torch.cat(all_img),
        "txt_emb": torch.cat(all_txt),
        "labels": torch.cat(all_lbl),
        "severity": torch.cat(all_sev),
    }


# ---------- retrieval metrics per class ----------

def compute_retrieval_metrics(img_emb, txt_emb, labels, classes):
    sim = img_emb @ txt_emb.T
    n = sim.shape[0]

    results = {"overall": {}, "per_class": {}}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            correct = sum(labels[i] in labels[top[i]] for i in range(n))
            results["overall"][f"{tag}_R@{k}"] = round(100.0 * correct / n, 2)

    for cls_idx, cls_name in enumerate(classes):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue

        cls_results = {"count": int(mask.sum())}
        indices = torch.where(mask)[0]

        for tag, s in [("I2T", sim), ("T2I", sim.T)]:
            for k in [1, 5, 10]:
                _, top = s.topk(k, dim=1)
                correct = sum(labels[indices[i]] in labels[top[indices[i]]] for i in range(len(indices)))
                cls_results[f"{tag}_R@{k}"] = round(100.0 * correct / len(indices), 2)

        results["per_class"][cls_name] = cls_results

    return results


# ---------- plot 1: I2T grid ----------

def plot_i2t_grid(dataset, img_emb, txt_emb, labels, classes):
    random.seed(42)

    class_samples = {c: [] for c in classes}
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        disease = row["disease"]
        if len(class_samples[disease]) < cfg.samples_per_class:
            disk = dataset._locate(row["image_path"])
            if disk:
                prompts = dataset.prompts[row["image_path"]]
                class_samples[disease].append({
                    "idx": idx, "path": disk, "disease": disease,
                    "prompt_a": prompts["a"], "prompt_b": prompts["b"],
                })
        if all(len(v) >= cfg.samples_per_class for v in class_samples.values()):
            break

    all_samples = []
    for cls in classes:
        all_samples.extend(class_samples[cls])

    n = len(all_samples)
    sim = img_emb @ txt_emb.T

    fig, axes = plt.subplots(n, 2, figsize=(18, 4 * n))

    for i, sample in enumerate(all_samples):
        idx = sample["idx"]
        disease = sample["disease"]

        img = Image.open(sample["path"]).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        axes[i, 0].imshow(np.array(img))
        axes[i, 0].set_title(f"Query: {disease}", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        scores, top_ids = sim[idx].topk(cfg.top_k)
        match_text = ""
        for rank, (score, tid) in enumerate(zip(scores, top_ids)):
            match_cls = classes[labels[tid].item()]
            correct = "✓" if match_cls == disease else "✗"
            row_match = dataset.df.iloc[tid.item()]
            prompts_match = dataset.prompts[row_match["image_path"]]

            match_text += f"#{rank+1} [{correct}] {match_cls} (sim={score.item():.3f})\n"
            match_text += f"  Structure: {prompts_match['a'][:80]}...\n"
            match_text += f"  Lesions: {prompts_match['b'][:80]}...\n\n"

        axes[i, 1].text(0.05, 0.95, match_text, transform=axes[i, 1].transAxes,
                        fontsize=9, verticalalignment="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        axes[i, 1].set_title(f"Top {cfg.top_k} Retrieved Texts", fontsize=12)
        axes[i, 1].axis("off")

    plt.suptitle("Image-to-Text Retrieval (I2T) — CrossAttention Fusion", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/i2t_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cfg.output_dir}/i2t_grid.png")


# ---------- plot 2: T2I grid ----------

def plot_t2i_grid(dataset, img_emb, txt_emb, labels, classes):
    random.seed(42)

    class_samples = {c: [] for c in classes}
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        row = dataset.df.iloc[idx]
        disease = row["disease"]
        if len(class_samples[disease]) < cfg.samples_per_class:
            disk = dataset._locate(row["image_path"])
            if disk:
                prompts = dataset.prompts[row["image_path"]]
                class_samples[disease].append({
                    "idx": idx, "path": disk, "disease": disease,
                    "prompt_a": prompts["a"], "prompt_b": prompts["b"],
                })
        if all(len(v) >= cfg.samples_per_class for v in class_samples.values()):
            break

    all_samples = []
    for cls in classes:
        all_samples.extend(class_samples[cls])

    n = len(all_samples)
    sim = txt_emb @ img_emb.T

    fig, axes = plt.subplots(n, cfg.top_k + 1, figsize=(5 * (cfg.top_k + 1), 4.5 * n))

    for i, sample in enumerate(all_samples):
        idx = sample["idx"]
        disease = sample["disease"]

        query_txt = f"[{disease}]\n{sample['prompt_a'][:60]}...\n{sample['prompt_b'][:60]}..."
        axes[i, 0].text(0.5, 0.5, query_txt, transform=axes[i, 0].transAxes,
                        fontsize=9, ha="center", va="center", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        axes[i, 0].set_title("Text Query", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        scores, top_ids = sim[idx].topk(cfg.top_k)
        for rank, (score, tid) in enumerate(zip(scores, top_ids)):
            match_row = dataset.df.iloc[tid.item()]
            match_cls = classes[labels[tid].item()]
            correct = "✓" if match_cls == disease else "✗"

            match_disk = dataset._locate(match_row["image_path"])
            if match_disk:
                match_img = Image.open(match_disk).convert("RGB")
                axes[i, rank + 1].imshow(np.array(match_img))
            else:
                axes[i, rank + 1].text(0.5, 0.5, "Not found", ha="center", va="center")

            color = "green" if match_cls == disease else "red"
            axes[i, rank + 1].set_title(
                f"#{rank+1} {correct} {match_cls}\nsim={score.item():.3f}",
                fontsize=10, color=color,
            )
            axes[i, rank + 1].axis("off")

    plt.suptitle("Text-to-Image Retrieval (T2I) — CrossAttention Fusion", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/t2i_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cfg.output_dir}/t2i_grid.png")


# ---------- plot 3: per-class R@K ----------

def plot_per_class_metrics(metrics, classes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

    for ax_idx, tag in enumerate(["I2T", "T2I"]):
        x = np.arange(3)
        width = 0.2

        for cls_idx, cls_name in enumerate(classes):
            if cls_name not in metrics["per_class"]:
                continue
            vals = [metrics["per_class"][cls_name].get(f"{tag}_R@{k}", 0) for k in [1, 5, 10]]
            axes[ax_idx].bar(x + cls_idx * width, vals, width,
                            label=cls_name, color=colors[cls_idx])

        axes[ax_idx].set_xticks(x + width * 1.5)
        axes[ax_idx].set_xticklabels(["R@1", "R@5", "R@10"])
        axes[ax_idx].set_ylabel("Retrieval %")
        axes[ax_idx].set_title(f"{tag} Retrieval per Class", fontsize=13)
        axes[ax_idx].legend()
        axes[ax_idx].grid(alpha=0.3, axis="y")
        axes[ax_idx].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/retrieval_per_class.png", dpi=150)
    plt.close()
    print(f"  Saved: {cfg.output_dir}/retrieval_per_class.png")


# ---------- plot 4: similarity distribution ----------

def plot_similarity_dist(img_emb, txt_emb, labels):
    sim = img_emb @ txt_emb.T
    n = sim.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    correct_sims = []
    incorrect_sims = []
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                correct_sims.append(sim[i, j].item())
            else:
                if random.random() < 0.1:
                    incorrect_sims.append(sim[i, j].item())

    ax.hist(correct_sims, bins=50, alpha=0.7, color="green",
            label=f"Same class (n={len(correct_sims)})", density=True)
    ax.hist(incorrect_sims, bins=50, alpha=0.7, color="red",
            label="Different class (sampled)", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Similarity Distribution: Same vs Different Class (CrossAttention Fusion)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/similarity_dist.png", dpi=150)
    plt.close()
    print(f"  Saved: {cfg.output_dir}/similarity_dist.png")


# ---------- plot 5: failure cases ----------

def plot_failures(dataset, img_emb, txt_emb, labels, classes):
    sim = img_emb @ txt_emb.T
    top1 = sim.argmax(dim=1)

    failures = []
    for i in range(len(labels)):
        pred_label = labels[top1[i]].item()
        true_label = labels[i].item()
        if pred_label != true_label:
            failures.append({
                "idx": i,
                "true": classes[true_label],
                "pred": classes[pred_label],
                "sim_correct": sim[i, i].item(),
                "sim_wrong": sim[i, top1[i]].item(),
            })

    if not failures:
        print("  No retrieval failures!")
        return

    show = failures[:min(8, len(failures))]
    n = len(show)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, f in enumerate(show):
        row = dataset.df.iloc[f["idx"]]
        disk = dataset._locate(row["image_path"])

        if disk:
            img = Image.open(disk).convert("RGB")
            axes[i].imshow(np.array(img))
        else:
            axes[i].text(0.5, 0.5, "Not found", ha="center")

        axes[i].set_title(
            f"True: {f['true']} -> Retrieved: {f['pred']} "
            f"(sim_correct={f['sim_correct']:.3f}, sim_wrong={f['sim_wrong']:.3f})",
            fontsize=11, color="red",
        )
        axes[i].axis("off")

    plt.suptitle(f"Retrieval Failures ({len(failures)} total errors out of {len(labels)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/failure_cases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {cfg.output_dir}/failure_cases.png")
    print(f"  Total failures: {len(failures)}/{len(labels)} ({100*len(failures)/len(labels):.1f}%)")


# ---------- main ----------

def main():
    set_seed()

    print(f"{'=' * 60}")
    print("  RETRIEVAL DEMO: MedSigLIP v3 (CrossAttention Fusion)")
    print(f"{'=' * 60}")

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)
    model.eval()
    print(f"  Model on {cfg.device}")

    dataset = OCT5kDataset(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=proc,
        mode="eval",
    )

    loader = DataLoader(dataset, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True,
                        collate_fn=collate_oct5k)

    data = extract_all(model, loader)
    print(f"  Extracted {len(data['img_emb'])} embeddings\n")

    print("  Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(data["img_emb"], data["txt_emb"],
                                        data["labels"], classes)

    print(f"\n  Overall:")
    for k, v in metrics["overall"].items():
        print(f"    {k}: {v}%")

    print(f"\n  Per class:")
    for cls_name, cls_metrics in metrics["per_class"].items():
        i2t = cls_metrics.get("I2T_R@1", 0)
        t2i = cls_metrics.get("T2I_R@1", 0)
        print(f"    {cls_name} ({cls_metrics['count']} samples): I2T R@1={i2t}% | T2I R@1={t2i}%")

    print(f"\n  Generating plots...")
    plot_i2t_grid(dataset, data["img_emb"], data["txt_emb"], data["labels"], classes)
    plot_t2i_grid(dataset, data["img_emb"], data["txt_emb"], data["labels"], classes)
    plot_per_class_metrics(metrics, classes)
    plot_similarity_dist(data["img_emb"], data["txt_emb"], data["labels"])
    plot_failures(dataset, data["img_emb"], data["txt_emb"], data["labels"], classes)

    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Results: {cfg.results_json}")

    print(f"\n{'=' * 60}")
    print(f"  Output: {cfg.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()