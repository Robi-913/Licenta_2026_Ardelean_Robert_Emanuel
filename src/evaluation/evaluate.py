"""
Step 4: Evaluare comparativa — CNN vs SigLIP vs MedSigLIP v3

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src.utils.seed import set_seed


def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------- config ----------

class Config:
    cnn_ckpt = "checkpoints/resnet18_final.pth"
    cnn_root = "data/old/raw"
    cnn_csv = "data/old/splits/val.csv"
    cnn_classes = 4
    cnn_size = 224

    sig_ckpt = "checkpoints/siglip_final.pth"
    sig_csv = "data/old/splits/val.csv"
    sig_root = "data/old/raw"
    sig_prompts = "data/old/prompts_expanded.json"
    sig_size = 224
    sig_patch = 16
    sig_img_dim = 384
    sig_img_depth = 6
    sig_img_heads = 6
    sig_vocab = 30522
    sig_max_len = 77
    sig_txt_dim = 256
    sig_txt_depth = 4
    sig_txt_heads = 4
    sig_out = 256

    med_model = "models/medsiglip-448"
    med_ckpt = "experiments/medsiglip_v3/ckpts/best.pth"
    med_csv = "data/oct5k/splits/test.csv"
    med_split_json = "data/oct5k/medgemma_prompts_split.json"
    med_sev_json = "data/oct5k/severity_scores.json"

    fig_dir = "experiments/figures/eval"
    results_json = "experiments/eval_results.json"

    bs = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.fig_dir, exist_ok=True)


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


# ---------- medsiglip v3 model ----------

class MedSigLIPMultiTask(nn.Module):

    def __init__(self, model_path, n_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        dim = self.backbone.config.vision_config.hidden_size

        # v3: fara Sigmoid
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

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        img_emb = self.encode_image(pixel_values)
        ea = self.encode_text(ids_a, mask_a)
        eb = self.encode_text(ids_b, mask_b)

        # v3: CrossAttentionFusion (nu mean merge)
        merged = self.fusion(ea, eb)

        # v3: clamp in loc de sigmoid
        sev = self.sev_head(img_emb).squeeze(-1).clamp(0, 1)
        cls = self.cls_head(img_emb)

        return img_emb, ea, eb, merged, self.logit_scale, sev, cls


# ---------- retrieval helper ----------

def compute_retrieval(img_emb, txt_emb, labels):
    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    out = {}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            hit = sum(labels[i] in labels[top[i]] for i in range(n))
            out[f"{tag}_R@{k}"] = round(100.0 * hit / n, 2)

    return out


# ---------- 1. CNN ----------

def eval_cnn():
    print(f"\n{'─' * 50}")
    print("  EVAL: CNN ResNet18 Baseline")
    print(f"{'─' * 50}")

    from src.models.cnn_resnet18 import ResNet18OCT
    from src.datasets.oct_dataset import OCTDataset, get_transforms

    if not os.path.exists(cfg.cnn_ckpt):
        print(f"  SKIP: {cfg.cnn_ckpt} not found")
        return None

    model = ResNet18OCT(num_classes=cfg.cnn_classes, use_pretrained=False)
    ckpt = torch.load(cfg.cnn_ckpt, map_location=cfg.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(cfg.device)
    model.eval()

    ds = OCTDataset(
        csv_path=cfg.cnn_csv,
        data_root=cfg.cnn_root,
        transform=get_transforms("eval", cfg.cnn_size),
        tokenizer=None,
        mode="eval",
    )

    def collate(batch):
        imgs = torch.stack([b["image"] for b in batch])
        lbls = torch.tensor([b["label"] for b in batch])
        return imgs, lbls

    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  CNN eval"):
            out = model(imgs.to(cfg.device))
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.numpy())

    p = np.array(preds_all)
    t = np.array(labels_all)

    acc = accuracy_score(t, p)
    f1 = f1_score(t, p, average="macro")
    report = classification_report(t, p, target_names=ds.classes, digits=4, output_dict=True)

    print(f"  Accuracy: {acc * 100:.1f}%")
    print(f"  F1 Macro: {f1:.4f}")

    cm = confusion_matrix(t, p)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ds.classes, yticklabels=ds.classes)
    plt.title("CNN ResNet18 - Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/cnn_cm.png", dpi=150)
    plt.close()

    clear_mem()

    return {
        "model": "CNN ResNet18",
        "dataset": "old (4 classes)",
        "accuracy": round(acc * 100, 2),
        "f1_macro": round(f1, 4),
        "classification_report": report,
    }


# ---------- 2. SigLIP ----------

def eval_siglip():
    print(f"\n{'─' * 50}")
    print("  EVAL: SigLIP (from scratch)")
    print(f"{'─' * 50}")

    from src.models.siglip_model import SigLIPModel
    from src.datasets.oct_dataset import OCTDataset, get_transforms
    from transformers import BertTokenizer

    if not os.path.exists(cfg.sig_ckpt):
        print(f"  SKIP: {cfg.sig_ckpt} not found")
        return None

    model = SigLIPModel(
        img_size=cfg.sig_size,
        patch_size=cfg.sig_patch,
        img_dim=cfg.sig_img_dim,
        img_depth=cfg.sig_img_depth,
        img_heads=cfg.sig_img_heads,
        vocab_size=cfg.sig_vocab,
        max_len=cfg.sig_max_len,
        txt_dim=cfg.sig_txt_dim,
        txt_depth=cfg.sig_txt_depth,
        txt_heads=cfg.sig_txt_heads,
        out_dim=cfg.sig_out,
    )

    state = torch.load(cfg.sig_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(cfg.device)
    model.eval()

    tok = BertTokenizer.from_pretrained("bert-base-uncased")

    ds = OCTDataset(
        csv_path=cfg.sig_csv,
        data_root=cfg.sig_root,
        prompts_path=cfg.sig_prompts,
        transform=get_transforms("eval", cfg.sig_size),
        tokenizer=tok,
        mode="eval",
    )

    def collate(batch):
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch]),
        }

    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate)

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

    metrics = compute_retrieval(img_emb, txt_emb, labels)
    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2

    print(f"  I2T R@1={metrics['I2T_R@1']:.1f}% R@5={metrics['I2T_R@5']:.1f}%")
    print(f"  T2I R@1={metrics['T2I_R@1']:.1f}% R@5={metrics['T2I_R@5']:.1f}%")
    print(f"  Avg R@1: {avg_r1:.1f}%")

    clear_mem()

    return {
        "model": "SigLIP (scratch)",
        "dataset": "old (4 classes)",
        "avg_R@1": round(avg_r1, 2),
        **metrics,
    }


# ---------- 3. MedSigLIP v3 ----------

def eval_medsiglip():
    print(f"\n{'─' * 50}")
    print("  EVAL: MedSigLIP v3 Multi-Task Pipeline")
    print(f"{'─' * 50}")

    from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k

    if not os.path.exists(cfg.med_ckpt):
        print(f"  SKIP: {cfg.med_ckpt} not found")
        return None

    proc = AutoProcessor.from_pretrained(cfg.med_model)

    ckpt = torch.load(cfg.med_ckpt, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.med_model, n_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)
    model.eval()

    ds = OCT5kDataset(
        split_csv=cfg.med_csv,
        split_json=cfg.med_split_json,
        severity_json=cfg.med_sev_json,
        processor=proc,
        mode="eval",
    )

    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate_oct5k)

    all_img, all_txt, all_lbl = [], [], []
    all_sp, all_st = [], []
    all_cp, all_ct = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  MedSigLIP eval"):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
            ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
            ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
            mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

            with autocast(cfg.device, enabled=cfg.amp):
                # v3: forward returneaza 7 valori
                ie, ea, eb, te, _, sp, cl = model(pv, ia, ma, ib, mb)

            all_img.append(ie.cpu())
            all_txt.append(te.cpu())  # te = merged din fusion
            all_lbl.append(batch["label"])
            all_sp.append(sp.cpu())
            all_st.append(batch["severity"])
            all_cp.append(cl.argmax(1).cpu())
            all_ct.append(batch["label"])

            del pv, ia, ma, ib, mb, ie, ea, eb, te, sp, cl

    clear_mem()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    metrics = compute_retrieval(img_emb, txt_emb, labels)
    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2

    sp_pct = torch.cat(all_sp) * 100
    st_pct = torch.cat(all_st) * 100
    sev_mae = (sp_pct - st_pct).abs().mean().item()

    cp = torch.cat(all_cp).numpy()
    ct = torch.cat(all_ct).numpy()
    cls_acc = accuracy_score(ct, cp)
    cls_f1 = f1_score(ct, cp, average="macro")
    cls_report = classification_report(ct, cp, target_names=classes, digits=4, output_dict=True)

    print(f"  I2T R@1={metrics['I2T_R@1']:.1f}% R@5={metrics['I2T_R@5']:.1f}%")
    print(f"  T2I R@1={metrics['T2I_R@1']:.1f}% R@5={metrics['T2I_R@5']:.1f}%")
    print(f"  Avg R@1: {avg_r1:.1f}%")
    print(f"  Cls Accuracy: {cls_acc * 100:.1f}% | F1: {cls_f1:.4f}")
    print(f"  Severity MAE: {sev_mae:.1f}%")

    cm = confusion_matrix(ct, cp)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=classes, yticklabels=classes)
    plt.title("MedSigLIP v3 - Classification Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/medsiglip_cm.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(st_pct.numpy(), sp_pct.numpy(), alpha=0.3, s=10)
    plt.plot([0, 100], [0, 100], "r--", label="Perfect")
    plt.xlabel("True Severity (%)")
    plt.ylabel("Predicted Severity (%)")
    plt.title(f"MedSigLIP v3 - Severity (MAE={sev_mae:.1f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/medsiglip_sev_scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, cname in enumerate(classes):
        mask = ct == i
        if mask.sum() == 0:
            continue
        ts = st_pct.numpy()[mask]
        ps = sp_pct.numpy()[mask]
        mae_c = np.abs(ts - ps).mean()
        plt.scatter(ts, ps, alpha=0.4, s=15, label=f"{cname} (MAE={mae_c:.1f}%)")

    plt.plot([0, 100], [0, 100], "r--", alpha=0.5)
    plt.xlabel("True Severity (%)")
    plt.ylabel("Predicted Severity (%)")
    plt.title("Severity per Disease")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/medsiglip_sev_per_class.png", dpi=150)
    plt.close()

    return {
        "model": "MedSigLIP v3",
        "dataset": "OCT5k (4 classes)",
        "accuracy": round(cls_acc * 100, 2),
        "f1_macro": round(cls_f1, 4),
        "avg_R@1": round(avg_r1, 2),
        "severity_mae": round(sev_mae, 2),
        "classification_report": cls_report,
        **metrics,
    }


# ---------- comparison plots ----------

def plot_comparison(results):
    valid = [r for r in results if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    acc_data = [(r["model"], r["accuracy"]) for r in valid if "accuracy" in r]
    if acc_data:
        names, vals = zip(*acc_data)
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = axes[0].bar(names, vals, color=colors[:len(names)])
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{v}%", ha="center", fontweight="bold")
        axes[0].set_ylabel("Accuracy %")
        axes[0].set_title("Classification Accuracy")
        axes[0].set_ylim(0, 105)
        axes[0].grid(alpha=0.3, axis="y")

    ret_data = [r for r in valid if "I2T_R@1" in r]
    if ret_data:
        x = np.arange(3)
        w = 0.35
        for i, r in enumerate(ret_data):
            vals = [(r.get(f"I2T_R@{k}", 0) + r.get(f"T2I_R@{k}", 0)) / 2 for k in [1, 5, 10]]
            axes[1].bar(x + i * w, vals, w, label=r["model"])
        axes[1].set_xticks(x + w / 2)
        axes[1].set_xticklabels(["R@1", "R@5", "R@10"])
        axes[1].set_ylabel("Avg Retrieval %")
        axes[1].set_title("Retrieval Performance")
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis="y")

    caps = {}
    for r in valid:
        caps[r["model"]] = [
            r.get("accuracy", 0),
            r.get("avg_R@1", 0),
            max(0, 100 - r.get("severity_mae", 100)),
        ]

    if caps:
        cap_labels = ["Classification", "Retrieval", "Severity"]
        x = np.arange(len(cap_labels))
        w = 0.25
        for i, (name, vals) in enumerate(caps.items()):
            axes[2].bar(x + i * w, vals, w, label=name)
        axes[2].set_xticks(x + w)
        axes[2].set_xticklabels(cap_labels)
        axes[2].set_ylabel("Score %")
        axes[2].set_title("Model Capabilities")
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/comparison.png", dpi=150)
    plt.close()
    print(f"\n  Comparison plot: {cfg.fig_dir}/comparison.png")


# ---------- main ----------

def main():
    set_seed()

    print(f"{'=' * 70}")
    print("  STEP 4: COMPARATIVE EVALUATION")
    print(f"{'=' * 70}")

    results = []

    r1 = eval_cnn()
    results.append(r1)
    clear_mem()

    r2 = eval_siglip()
    results.append(r2)
    clear_mem()

    r3 = eval_medsiglip()
    results.append(r3)
    clear_mem()

    good = [r for r in results if r is not None]

    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(good, f, indent=2, ensure_ascii=False, default=str)

    plot_comparison(good)

    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for r in good:
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

    print(f"\n{'─' * 70}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'R@1':>8} {'Sev MAE':>10}")
    print(f"{'─' * 70}")
    for r in good:
        acc = f"{r['accuracy']}%" if "accuracy" in r else "-"
        r1_val = f"{r['avg_R@1']}%" if "avg_R@1" in r else "-"
        sev = f"{r['severity_mae']}%" if "severity_mae" in r else "-"
        print(f"  {r['model']:<25} {acc:>10} {r1_val:>8} {sev:>10}")
    print(f"{'─' * 70}")

    print(f"\n  Results: {cfg.results_json}")
    print(f"  Figures: {cfg.fig_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()