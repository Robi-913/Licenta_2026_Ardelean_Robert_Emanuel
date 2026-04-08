"""
Step 4: Evaluare comparativă — CNN Baseline vs SigLIP vs MedSigLIP Pipeline

Evaluează fiecare model pe test set-ul său:
  - CNN ResNet18:  data/old/splits/val.csv      → Accuracy, F1, Confusion Matrix
  - SigLIP (scratch): data/old/splits/val.csv   → R@1, R@5, R@10, Zero-shot Acc
  - MedSigLIP:     data/oct5k/splits/test.csv   → R@1, R@5, R@10, Zero-shot Acc

Output:
  experiments/figures/          ← plots comparative
  experiments/eval_results.json ← toate metricile

Rulare:
    python -m src.pipelines.medsiglip.evaluate
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, BertTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils.seed import set_seed, SEED


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
    medsiglip_prompts = "data/oct5k/medgemma_prompts.json"

    # output
    figures_dir = "experiments/figures"
    results_path = "experiments/eval_results.json"

    # general
    bs = 32
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.figures_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. EVAL CNN BASELINE
# ═══════════════════════════════════════════════════════════════════════

def eval_cnn():
    """Evaluează CNN ResNet18 pe test set."""
    print(f"\n{'─' * 50}")
    print("  EVAL: CNN ResNet18 Baseline")
    print(f"{'─' * 50}")

    from src.models.cnn_resnet18 import ResNet18OCT
    from src.datasets.oct_dataset import OCTDataset, get_transforms

    if not os.path.exists(cfg.cnn_checkpoint):
        print(f"  SKIP: checkpoint nu există ({cfg.cnn_checkpoint})")
        return None

    # model
    model = ResNet18OCT(num_classes=cfg.cnn_num_classes, use_pretrained=False)
    ckpt = torch.load(cfg.cnn_checkpoint, map_location=cfg.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(cfg.device)
    model.eval()

    # dataset
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

    loader = DataLoader(
        test_ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate,
    )

    # inference
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
    report = classification_report(
        labels, preds, target_names=test_ds.classes, digits=4, output_dict=True,
    )

    print(f"  Accuracy: {acc * 100:.1f}%")
    print(f"  F1 Macro: {f1:.4f}")

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.title("CNN ResNet18 — Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/cnn_confusion_matrix.png", dpi=150)
    plt.close()

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
    """Evaluează SigLIP-ul antrenat de la zero pe test set."""
    print(f"\n{'─' * 50}")
    print("  EVAL: SigLIP (from scratch)")
    print(f"{'─' * 50}")

    from src.models.siglip_model import SigLIPModel
    from src.datasets.oct_dataset import OCTDataset, get_transforms

    if not os.path.exists(cfg.siglip_checkpoint):
        print(f"  SKIP: checkpoint nu există ({cfg.siglip_checkpoint})")
        return None

    # model
    model = SigLIPModel(
        img_size=cfg.siglip_img_size,
        patch_size=cfg.siglip_patch_size,
        img_dim=cfg.siglip_img_dim,
        img_depth=cfg.siglip_img_depth,
        img_heads=cfg.siglip_img_heads,
        vocab_size=cfg.siglip_vocab_size,
        max_len=cfg.siglip_max_len,
        txt_dim=cfg.siglip_txt_dim,
        txt_depth=cfg.siglip_txt_depth,
        txt_heads=cfg.siglip_txt_heads,
        out_dim=cfg.siglip_out_dim,
    )

    state = torch.load(cfg.siglip_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(cfg.device)
    model.eval()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # dataset
    test_ds = OCTDataset(
        csv_path=cfg.siglip_test_csv,
        data_root=cfg.siglip_data_root,
        prompts_path=cfg.siglip_prompts,
        transform=get_transforms("eval", cfg.siglip_img_size),
        tokenizer=tokenizer,
        mode="eval",
    )

    def collate(batch):
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }

    loader = DataLoader(
        test_ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate,
    )

    # retrieval
    all_img, all_txt, all_lbl = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  SigLIP eval"):
            with autocast(cfg.device, enabled=cfg.amp):
                ie = model.encode_image(batch["images"].to(cfg.device))
                te = model.encode_text(
                    batch["input_ids"].to(cfg.device),
                    batch["attention_mask"].to(cfg.device),
                )
            all_img.append(ie.cpu())
            all_txt.append(te.cpu())
            all_lbl.append(batch["labels"])

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
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
    """Evaluează MedSigLIP fine-tunat pe test set OCT5k."""
    print(f"\n{'─' * 50}")
    print("  EVAL: MedSigLIP Pipeline (MedGemma + MedSigLIP)")
    print(f"{'─' * 50}")

    from src.datasets.oct5k_medsiglip import OCT5kMedSigLIP, collate_medsiglip
    from src.training.train_medsiglip import MedSigLIPContrastive

    if not os.path.exists(cfg.medsiglip_checkpoint):
        print(f"  SKIP: checkpoint nu există ({cfg.medsiglip_checkpoint})")
        return None

    # processor
    processor = AutoProcessor.from_pretrained(cfg.medsiglip_model_path)

    # model
    model = MedSigLIPContrastive(cfg.medsiglip_model_path)
    ckpt = torch.load(cfg.medsiglip_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model = model.to(cfg.device)
    model.eval()

    # dataset
    test_ds = OCT5kMedSigLIP(
        split_csv=cfg.medsiglip_test_csv,
        prompts_json=cfg.medsiglip_prompts,
        processor=processor,
        mode="eval",
    )

    loader = DataLoader(
        test_ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate_medsiglip,
    )

    # retrieval + classification
    all_img, all_txt, all_lbl = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  MedSigLIP eval"):
            pv = batch["pixel_values"].to(cfg.device)
            ids = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)

            with autocast(cfg.device, enabled=cfg.amp):
                ie = model.encode_image(pv)
                te = model.encode_text(ids, mask)

            all_img.append(ie.cpu())
            all_txt.append(te.cpu())
            all_lbl.append(batch["label"])

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
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

    # zero-shot classification: pt fiecare imagine, clasa = clasa textului cu cel mai mare scor
    i2t_preds = sim.argmax(dim=1).numpy()
    zs_preds = labels[i2t_preds].numpy() if len(labels) > 0 else np.array([])
    zs_labels = labels.numpy()

    zs_acc = accuracy_score(zs_labels, zs_preds) if len(zs_preds) > 0 else 0.0
    zs_f1 = f1_score(zs_labels, zs_preds, average="macro") if len(zs_preds) > 0 else 0.0

    print(f"  I2T R@1={metrics['I2T_R@1']:.1f}% R@5={metrics['I2T_R@5']:.1f}%")
    print(f"  T2I R@1={metrics['T2I_R@1']:.1f}% R@5={metrics['T2I_R@5']:.1f}%")
    print(f"  Avg R@1: {avg_r1:.1f}%")
    print(f"  Zero-shot Acc: {zs_acc * 100:.1f}% | F1: {zs_f1:.4f}")

    # confusion matrix pt zero-shot
    if len(zs_preds) > 0:
        class_names = test_ds.classes
        cm = confusion_matrix(zs_labels, zs_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("MedSigLIP Pipeline — Zero-shot Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f"{cfg.figures_dir}/medsiglip_confusion_matrix.png", dpi=150)
        plt.close()

    return {
        "model": "MedSigLIP Pipeline",
        "dataset": "OCT5k (4 classes)",
        "avg_R@1": round(avg_r1, 2),
        "zero_shot_accuracy": round(zs_acc * 100, 2),
        "zero_shot_f1": round(zs_f1, 4),
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# COMPARATIVE PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(results):
    """Generează grafic comparativ între modele."""
    models = [r["model"] for r in results if r is not None]
    if len(models) == 0:
        print("  Nu sunt rezultate de comparat.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # accuracy / F1 comparison
    acc_data = []
    for r in results:
        if r is None:
            continue
        name = r["model"]
        if "accuracy" in r:
            acc_data.append({"Model": name, "Metric": "Accuracy", "Value": r["accuracy"]})
            acc_data.append({"Model": name, "Metric": "F1 Macro", "Value": r["f1_macro"] * 100})
        if "zero_shot_accuracy" in r:
            acc_data.append({"Model": name, "Metric": "Accuracy", "Value": r["zero_shot_accuracy"]})
            acc_data.append({"Model": name, "Metric": "F1 Macro", "Value": r["zero_shot_f1"] * 100})

    if acc_data:
        df_acc = pd.DataFrame(acc_data)
        colors = {"Accuracy": "#4C72B0", "F1 Macro": "#DD8452"}
        x = np.arange(len(df_acc["Model"].unique()))
        width = 0.35

        models_unique = df_acc["Model"].unique()
        for i, metric in enumerate(["Accuracy", "F1 Macro"]):
            vals = [df_acc[(df_acc["Model"] == m) & (df_acc["Metric"] == metric)]["Value"].values
                    for m in models_unique]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            axes[0].bar(x + i * width, vals, width, label=metric, color=colors[metric])

        axes[0].set_xticks(x + width / 2)
        axes[0].set_xticklabels(models_unique, rotation=15, ha="right")
        axes[0].set_ylabel("%")
        axes[0].set_title("Classification Performance")
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis="y")

    # retrieval R@K comparison
    retrieval_data = []
    for r in results:
        if r is None:
            continue
        name = r["model"]
        if "I2T_R@1" in r:
            for k in [1, 5, 10]:
                avg_rk = (r.get(f"I2T_R@{k}", 0) + r.get(f"T2I_R@{k}", 0)) / 2
                retrieval_data.append({"Model": name, "K": f"R@{k}", "Value": avg_rk})

    if retrieval_data:
        df_ret = pd.DataFrame(retrieval_data)
        models_ret = df_ret["Model"].unique()
        x = np.arange(3)
        width = 0.35
        for i, m in enumerate(models_ret):
            vals = [df_ret[(df_ret["Model"] == m) & (df_ret["K"] == f"R@{k}")]["Value"].values[0]
                    for k in [1, 5, 10]]
            axes[1].bar(x + i * width, vals, width, label=m)

        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels(["R@1", "R@5", "R@10"])
        axes[1].set_ylabel("Avg Retrieval %")
        axes[1].set_title("Retrieval Performance (Avg I2T + T2I)")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{cfg.figures_dir}/model_comparison.png", dpi=150)
    plt.close()
    print(f"\n  Plot comparativ salvat: {cfg.figures_dir}/model_comparison.png")


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

    # 2. SigLIP
    siglip_result = eval_siglip()
    results.append(siglip_result)

    # 3. MedSigLIP
    medsiglip_result = eval_medsiglip()
    results.append(medsiglip_result)

    # salvăm rezultatele
    valid_results = [r for r in results if r is not None]
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)

    # plot comparativ
    plot_comparison(valid_results)

    # raport final
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    for r in valid_results:
        print(f"\n  {r['model']} ({r['dataset']}):")
        if "accuracy" in r:
            print(f"    Accuracy: {r['accuracy']}%")
            print(f"    F1 Macro: {r['f1_macro']}")
        if "avg_R@1" in r:
            print(f"    Avg R@1:  {r['avg_R@1']}%")
        if "zero_shot_accuracy" in r:
            print(f"    ZS Acc:   {r['zero_shot_accuracy']}%")
            print(f"    ZS F1:    {r['zero_shot_f1']}")
        if "I2T_R@1" in r:
            print(f"    I2T: R@1={r['I2T_R@1']}% R@5={r['I2T_R@5']}% R@10={r['I2T_R@10']}%")
            print(f"    T2I: R@1={r['T2I_R@1']}% R@5={r['T2I_R@5']}% R@10={r['T2I_R@10']}%")

    print(f"\n  Rezultate salvate: {cfg.results_path}")
    print(f"  Figuri: {cfg.figures_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()