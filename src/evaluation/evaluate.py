"""
Step 4: Evaluare comparativă — CNN Baseline vs SigLIP vs MedSigLIP Pipeline

Evaluează fiecare model pe test set-ul său:
  - CNN ResNet18:     → Accuracy, F1, Confusion Matrix
  - SigLIP (scratch): → R@1, R@5, R@10
  - MedSigLIP:        → R@1, R@5, R@10, Classification, Severity MAE

Output:
  experiments/figures/          ← plots comparative
  experiments/eval_results.json ← toate metricile

Rulare:
    python -m src.evaluation.evaluate
"""

import os
import sys
import json
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


from src.utils.seed import set_seed


def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # CNN
    cnn_checkpoint = "checkpoints/resnet18_final.pth"
    cnn_data_root = "data/old/raw"
    cnn_test_csv = "data/old/splits/val.csv"
    cnn_num_classes = 4
    cnn_img_size = 224

    # SigLIP (scratch)
    siglip_checkpoint = "checkpoints/siglip_final.pth"
    siglip_test_csv = "data/old/splits/val.csv"
    siglip_data_root = "data/old/raw"
    siglip_prompts = "data/old/prompts_expanded.json"
    siglip_img_size = 224
    siglip_patch_size = 16
    siglip_img_dim = 384
    siglip_img_depth = 6
    siglip_img_heads = 6
    siglip_vocab_size = 30522
    siglip_max_len = 77
    siglip_txt_dim = 256
    siglip_txt_depth = 4
    siglip_txt_heads = 4
    siglip_out_dim = 256

    # MedSigLIP
    medsiglip_model_path = "models/medsiglip-448"
    medsiglip_checkpoint = "experiments/medsiglip_pipeline/ckpts/best.pth"
    medsiglip_test_csv = "data/oct5k/splits/test.csv"
    medsiglip_split_json = "data/oct5k/medgemma_prompts_split.json"
    medsiglip_severity_json = "data/oct5k/severity_scores.json"

    # output
    figures_dir = "experiments/figures"
    results_path = "experiments/eval_results.json"

    # general
    bs = 8  # mic pt VRAM safety
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.figures_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MEDSIGLIP MODEL (copie din train pt a nu avea import circular)
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

    def encode_text(self, input_ids, attention_mask):
        out = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        img_emb = self.encode_image(pixel_values)
        emb_a = self.encode_text(ids_a, mask_a)
        emb_b = self.encode_text(ids_b, mask_b)
        merged_txt = F.normalize((emb_a + emb_b) / 2, p=2, dim=-1)
        sev_pred = self.severity_head(img_emb).squeeze(-1)
        cls_logits = self.cls_head(img_emb)
        return img_emb, merged_txt, self.logit_scale, sev_pred, cls_logits


# ═══════════════════════════════════════════════════════════════════════
# 1. EVAL CNN BASELINE
# ═══════════════════════════════════════════════════════════════════════

def eval_cnn():
    print(f"\n{'─' * 50}")
    print("  EVAL: CNN ResNet18 Baseline")
    print(f"{'─' * 50}")

    from src.models.cnn_resnet18 import ResNet18OCT
    from src.datasets.oct_dataset import OCTDataset, get_transforms

    if not os.path.exists(cfg.cnn_checkpoint):
        print(f"  SKIP: checkpoint nu exista ({cfg.cnn_checkpoint})")
        return None

    model = ResNet18OCT(num_classes=cfg.cnn_num_classes, use_pretrained=False)
    ckpt = torch.load(cfg.cnn_checkpoint, map_location=cfg.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(cfg.device)
    model.eval()

    test_ds = OCTDataset(
        csv_path=cfg.cnn_test_csv,
        data_root=cfg.cnn_data_root,
        transform=get_transforms("eval", cfg.cnn_img_size),
        tokenizer=None,
        mode="eval",
    )

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch])
        return images, labels

    loader = DataLoader(test_ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  CNN eval"):
            imgs = imgs.to(cfg.device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, target_names=test_ds.classes, digits=4, output_dict=True)

    print(f"  Accuracy: {acc * 100:.1f}%")
    print(f"  F1 Macro: {f1:.4f}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.title("CNN ResNet18 — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/cnn_confusion_matrix.png", dpi=150)
    plt.close()

    free_memory()

    return {
        "model": "CNN ResNet18",
        "dataset": "old (4 classes)",
        "accuracy": round(acc * 100, 2),
        "f1_macro": round(f1, 4),
        "classification_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. EVAL SIGLIP (SCRATCH)
# ═══════════════════════════════════════════════════════════════════════

def eval_siglip():
    print(f"\n{'─' * 50}")
    print("  EVAL: SigLIP (from scratch)")
    print(f"{'─' * 50}")

    from src.models.siglip_model import SigLIPModel
    from src.datasets.oct_dataset import OCTDataset, get_transforms
    from transformers import BertTokenizer

    if not os.path.exists(cfg.siglip_checkpoint):
        print(f"  SKIP: checkpoint nu exista ({cfg.siglip_checkpoint})")
        return None

    model = SigLIPModel(
        img_size=cfg.siglip_img_size, patch_size=cfg.siglip_patch_size,
        img_dim=cfg.siglip_img_dim, img_depth=cfg.siglip_img_depth,
        img_heads=cfg.siglip_img_heads, vocab_size=cfg.siglip_vocab_size,
        max_len=cfg.siglip_max_len, txt_dim=cfg.siglip_txt_dim,
        txt_depth=cfg.siglip_txt_depth, txt_heads=cfg.siglip_txt_heads,
        out_dim=cfg.siglip_out_dim,
    )

    state = torch.load(cfg.siglip_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(cfg.device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_ds = OCTDataset(
        csv_path=cfg.siglip_test_csv, data_root=cfg.siglip_data_root,
        prompts_path=cfg.siglip_prompts,
        transform=get_transforms("eval", cfg.siglip_img_size),
        tokenizer=tokenizer, mode="eval",
    )

    def collate(batch):
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }

    loader = DataLoader(test_ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate)

    all_img, all_txt, all_lbl = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  SigLIP eval"):
            with autocast(cfg.device, enabled=cfg.amp):
                ie = model.encode_image(batch["images"].to(cfg.device))
                te = model.encode_text(batch["input_ids"].to(cfg.device),
                                       batch["attention_mask"].to(cfg.device))
            all_img.append(ie.cpu()); all_txt.append(te.cpu())
            all_lbl.append(batch["labels"])

    img_emb = torch.cat(all_img); txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    metrics = {}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            correct = sum(labels[i] in labels[top[i]] for i in range(n))
            metrics[f"{tag}_R@{k}"] = round(100.0 * correct / n, 2)

    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2

    print(f"  I2T R@1={metrics['I2T_R@1']:.1f}% R@5={metrics['I2T_R@5']:.1f}%")
    print(f"  T2I R@1={metrics['T2I_R@1']:.1f}% R@5={metrics['T2I_R@5']:.1f}%")
    print(f"  Avg R@1: {avg_r1:.1f}%")

    free_memory()

    return {
        "model": "SigLIP (scratch)",
        "dataset": "old (4 classes)",
        "avg_R@1": round(avg_r1, 2),
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. EVAL MEDSIGLIP PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def eval_medsiglip():
    print(f"\n{'─' * 50}")
    print("  EVAL: MedSigLIP Multi-Task Pipeline")
    print(f"{'─' * 50}")

    from src.datasets.oct5k_medsiglip import OCT5kMedSigLIP, collate_medsiglip

    if not os.path.exists(cfg.medsiglip_checkpoint):
        print(f"  SKIP: checkpoint nu exista ({cfg.medsiglip_checkpoint})")
        return None

    processor = AutoProcessor.from_pretrained(cfg.medsiglip_model_path)

    # incarcam checkpoint pt a vedea num_classes
    ckpt = torch.load(cfg.medsiglip_checkpoint, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.medsiglip_model_path, num_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)
    model.eval()

    # dataset cu dual prompts + severity
    test_ds = OCT5kMedSigLIP(
        split_csv=cfg.medsiglip_test_csv,
        split_json=cfg.medsiglip_split_json,
        severity_json=cfg.medsiglip_severity_json,
        processor=processor,
        mode="eval",
    )

    loader = DataLoader(test_ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate_medsiglip)

    all_img, all_txt, all_lbl = [], [], []
    all_sev_pred, all_sev_true = [], []
    all_cls_pred, all_cls_true = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  MedSigLIP eval"):
            pv = batch["pixel_values"].to(cfg.device)
            ids_a = batch["input_ids_a"].to(cfg.device)
            mask_a = batch["attention_mask_a"].to(cfg.device)
            ids_b = batch["input_ids_b"].to(cfg.device)
            mask_b = batch["attention_mask_b"].to(cfg.device)

            with autocast(cfg.device, enabled=cfg.amp):
                ie, te, _, sp, cl = model(pv, ids_a, mask_a, ids_b, mask_b)

            all_img.append(ie.cpu()); all_txt.append(te.cpu())
            all_lbl.append(batch["label"])
            all_sev_pred.append(sp.cpu()); all_sev_true.append(batch["severity"])
            all_cls_pred.append(cl.argmax(1).cpu()); all_cls_true.append(batch["label"])

            del pv, ids_a, mask_a, ids_b, mask_b, ie, te, sp, cl

    free_memory()

    img_emb = torch.cat(all_img); txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    metrics = {}

    # retrieval R@K
    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            correct = sum(labels[i] in labels[top[i]] for i in range(n))
            metrics[f"{tag}_R@{k}"] = round(100.0 * correct / n, 2)

    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2

    # severity MAE
    sev_p = torch.cat(all_sev_pred) * 100
    sev_t = torch.cat(all_sev_true) * 100
    sev_mae = (sev_p - sev_t).abs().mean().item()

    # classification accuracy + F1
    cls_pred = torch.cat(all_cls_pred).numpy()
    cls_true = torch.cat(all_cls_true).numpy()
    cls_acc = accuracy_score(cls_true, cls_pred)
    cls_f1 = f1_score(cls_true, cls_pred, average="macro")
    cls_report = classification_report(cls_true, cls_pred, target_names=classes, digits=4, output_dict=True)

    print(f"  I2T R@1={metrics['I2T_R@1']:.1f}% R@5={metrics['I2T_R@5']:.1f}%")
    print(f"  T2I R@1={metrics['T2I_R@1']:.1f}% R@5={metrics['T2I_R@5']:.1f}%")
    print(f"  Avg R@1: {avg_r1:.1f}%")
    print(f"  Cls Accuracy: {cls_acc * 100:.1f}% | F1: {cls_f1:.4f}")
    print(f"  Severity MAE: {sev_mae:.1f}%")

    # confusion matrix - clasificare
    cm = confusion_matrix(cls_true, cls_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=classes)
    plt.title("MedSigLIP Multi-Task — Classification Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/medsiglip_confusion_matrix.png", dpi=150)
    plt.close()

    # severity scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(sev_t.numpy(), sev_p.numpy(), alpha=0.3, s=10)
    plt.plot([0, 100], [0, 100], "r--", label="Perfect")
    plt.xlabel("True Severity (%)")
    plt.ylabel("Predicted Severity (%)")
    plt.title(f"MedSigLIP — Severity Prediction (MAE={sev_mae:.1f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/medsiglip_severity_scatter.png", dpi=150)
    plt.close()

    # severity per disease category
    plt.figure(figsize=(10, 6))
    for i, cls_name in enumerate(classes):
        mask = cls_true == i
        if mask.sum() > 0:
            true_s = sev_t.numpy()[mask]
            pred_s = sev_p.numpy()[mask]
            mae_cls = np.abs(true_s - pred_s).mean()
            plt.scatter(true_s, pred_s, alpha=0.4, s=15, label=f"{cls_name} (MAE={mae_cls:.1f}%)")

    plt.plot([0, 100], [0, 100], "r--", alpha=0.5)
    plt.xlabel("True Severity (%)")
    plt.ylabel("Predicted Severity (%)")
    plt.title("Severity per Disease Category")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/medsiglip_severity_per_class.png", dpi=150)
    plt.close()

    return {
        "model": "MedSigLIP Multi-Task",
        "dataset": "OCT5k (4 classes)",
        "accuracy": round(cls_acc * 100, 2),
        "f1_macro": round(cls_f1, 4),
        "avg_R@1": round(avg_r1, 2),
        "severity_mae": round(sev_mae, 2),
        "classification_report": cls_report,
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# COMPARATIVE PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(results):
    models = [r["model"] for r in results if r is not None]
    if len(models) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Classification Accuracy comparison
    acc_models, acc_vals = [], []
    for r in results:
        if r and "accuracy" in r:
            acc_models.append(r["model"])
            acc_vals.append(r["accuracy"])

    if acc_models:
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = axes[0].bar(acc_models, acc_vals, color=colors[:len(acc_models)])
        for bar, val in zip(bars, acc_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val}%", ha="center", fontweight="bold")
        axes[0].set_ylabel("Accuracy %")
        axes[0].set_title("Classification Accuracy")
        axes[0].set_ylim(0, 105)
        axes[0].grid(alpha=0.3, axis="y")

    # 2. Retrieval R@K comparison
    retrieval_models = []
    for r in results:
        if r and "I2T_R@1" in r:
            retrieval_models.append(r)

    if retrieval_models:
        x = np.arange(3)
        width = 0.35
        for i, r in enumerate(retrieval_models):
            vals = [(r.get(f"I2T_R@{k}", 0) + r.get(f"T2I_R@{k}", 0)) / 2 for k in [1, 5, 10]]
            bars = axes[1].bar(x + i * width, vals, width, label=r["model"])
        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels(["R@1", "R@5", "R@10"])
        axes[1].set_ylabel("Avg Retrieval %")
        axes[1].set_title("Retrieval Performance")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")

    # 3. Capabilities comparison (radar-style as bar)
    capabilities = ["Classification", "Retrieval", "Severity"]
    model_caps = {}
    for r in results:
        if r is None:
            continue
        name = r["model"]
        model_caps[name] = [
            r.get("accuracy", 0),
            r.get("avg_R@1", 0),
            max(0, 100 - r.get("severity_mae", 100)),  # inverted: higher = better
        ]

    if model_caps:
        x = np.arange(len(capabilities))
        width = 0.25
        for i, (name, vals) in enumerate(model_caps.items()):
            axes[2].bar(x + i * width, vals, width, label=name)
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(capabilities)
        axes[2].set_ylabel("Score %")
        axes[2].set_title("Model Capabilities")
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/model_comparison.png", dpi=150)
    plt.close()
    print(f"\n  Plot comparativ: {cfg.figures_dir}/model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    print(f"{'=' * 70}")
    print("  STEP 4: COMPARATIVE EVALUATION")
    print(f"{'=' * 70}")

    results = []

    # 1. CNN
    cnn_result = eval_cnn()
    results.append(cnn_result)
    free_memory()

    # 2. SigLIP
    siglip_result = eval_siglip()
    results.append(siglip_result)
    free_memory()

    # 3. MedSigLIP
    medsiglip_result = eval_medsiglip()
    results.append(medsiglip_result)
    free_memory()

    # salvam rezultatele
    valid_results = [r for r in results if r is not None]
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False, default=str)

    # plot comparativ
    plot_comparison(valid_results)

    # raport final
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for r in valid_results:
        print(f"\n  {r['model']} ({r['dataset']}):")
        if "accuracy" in r:
            print(f"    Accuracy:     {r['accuracy']}%")
        if "f1_macro" in r:
            print(f"    F1 Macro:     {r['f1_macro']}")
        if "avg_R@1" in r:
            print(f"    Avg R@1:      {r['avg_R@1']}%")
        if "severity_mae" in r:
            print(f"    Severity MAE: {r['severity_mae']}%")
        if "I2T_R@1" in r:
            print(f"    I2T: R@1={r['I2T_R@1']}% R@5={r['I2T_R@5']}% R@10={r['I2T_R@10']}%")
            print(f"    T2I: R@1={r['T2I_R@1']}% R@5={r['T2I_R@5']}% R@10={r['T2I_R@10']}%")

    # tabel comparativ final
    print(f"\n{'─' * 70}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'R@1':>8} {'Sev MAE':>10} {'Multi-task':>12}")
    print(f"{'─' * 70}")
    for r in valid_results:
        acc = f"{r.get('accuracy', '—')}%" if "accuracy" in r else "—"
        r1 = f"{r.get('avg_R@1', '—')}%" if "avg_R@1" in r else "—"
        sev = f"{r.get('severity_mae', '—')}%" if "severity_mae" in r else "—"
        mt = "✅" if ("accuracy" in r and "avg_R@1" in r and "severity_mae" in r) else "❌"
        print(f"  {r['model']:<25} {acc:>10} {r1:>8} {sev:>10} {mt:>12}")
    print(f"{'─' * 70}")

    print(f"\n  Rezultate: {cfg.results_path}")
    print(f"  Figuri: {cfg.figures_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()