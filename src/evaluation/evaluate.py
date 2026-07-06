"""
evaluate.py — Evaluare comparativa: CNN vs MedSigLIP pretrained vs MedSigLIP v13

Compara:
  1. CNN ResNet18          — baseline clasificare, antrenat pe splits_v3
  2. MedSigLIP pretrained  — backbone pur fara fine-tuning, zero-shot
  3. MedSigLIP v13         — fine-tuned multi-task (retrieval + clasificare + severitate)

Rulare:
    python -m src.evaluation.evaluate
"""

import gc
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


# CONFIG

class Config:
    # CNN ResNet18 — antrenat pe splits_v3
    cnn_ckpt    = "checkpoints/resnet18_v2_final.pth"
    cnn_classes = 4
    cnn_size    = 224

    # MedSigLIP — acelasi model path pt ambele variante
    med_model      = "models/medsiglip-448"
    med_ckpt       = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"
    med_csv        = "data/oct5k/splits_v3/test.csv"
    med_split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    med_sev_json   = "data/oct5k/severity_scores_v2.json"

    fig_dir      = "experiments/figures/eval"
    results_json = "experiments/eval_results_v13.json"

    batch_size = 8
    workers    = 0
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp    = torch.cuda.is_available()


cfg = Config()
os.makedirs(cfg.fig_dir, exist_ok=True)

CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]


# UTILITARE

def _free_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def compute_retrieval_metrics(img_emb: torch.Tensor, txt_emb: torch.Tensor, labels: torch.Tensor) -> dict:
    sim = img_emb @ txt_emb.T
    n   = sim.shape[0]
    out = {}
    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            hits = sum(labels[i] in labels[top[i]] for i in range(n))
            out[f"{tag}_R@{k}"] = round(100.0 * hits / n, 2)
    return out


def _save_confusion_matrix(cm, classes, title, path, cmap="Blues"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _make_oct5k_loader():
    proc = AutoProcessor.from_pretrained(cfg.med_model)
    ds   = OCT5kDataset(
        split_csv=cfg.med_csv, split_json=cfg.med_split_json,
        severity_json=cfg.med_sev_json, processor=proc, mode="eval",
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate_oct5k)
    return ds, loader, proc


# KEY REMAPPING — cheile vechi din checkpoint -> numele noi din clasa

def _remap_checkpoint_keys(state_dict: dict) -> dict:
    """
    Checkpointul salvat cu versiunea veche a modelului are nume diferite:
      sev_head       -> severity_head
      cls_head       -> classification_head
      fusion.attn_a2b -> fusion.attn_a_to_b
      fusion.attn_b2a -> fusion.attn_b_to_a
    Le remapam pt a fi compatibile cu clasa curenta.
    """
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace("sev_head.",        "severity_head.")
        k = k.replace("cls_head.",        "classification_head.")
        k = k.replace("fusion.attn_a2b.", "fusion.attn_a_to_b.")
        k = k.replace("fusion.attn_b2a.", "fusion.attn_b_to_a.")
        remapped[k] = v
    return remapped


# 1. CNN ResNet18

def eval_cnn() -> dict | None:
    print("\n  EVAL: CNN ResNet18 (splits_v3 test)")

    if not os.path.exists(cfg.cnn_ckpt):
        print(f"  SKIP: {cfg.cnn_ckpt} not found")
        return None

    from pathlib import Path

    import pandas as pd
    from PIL import Image, ImageFilter
    from torch.utils.data import Dataset
    from torchvision import transforms

    from src.model.cnn_resnet18 import ResNet18OCT

    _img_dirs = [
        "data/OCT5k/Images/Images_Automatic",
        "data/OCT5k/Images/Images_Manual",
        "data/OCT5k/Detection/Images",
    ]

    class CNNDataset(Dataset):
        def __init__(self):
            df = pd.read_csv(cfg.med_csv)
            self.classes = sorted(df["disease"].unique())
            self.lbl_map = {c: i for i, c in enumerate(self.classes)}
            self.tf = transforms.Compose([
                transforms.Resize((cfg.cnn_size, cfg.cnn_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.samples = []
            for _, row in df.iterrows():
                norm = row["image_path"].replace("\\", "/")
                for base in _img_dirs:
                    full = Path(base) / norm
                    found = None
                    if full.exists():
                        found = str(full)
                    else:
                        for ext in [".png", ".jpeg", ".jpg"]:
                            if full.with_suffix(ext).exists():
                                found = str(full.with_suffix(ext))
                                break
                    if found:
                        self.samples.append({"path": found, "label": self.lbl_map[row["disease"]]})
                        break

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            s   = self.samples[idx]
            img = Image.open(s["path"]).convert("RGB")
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            return {"image": self.tf(img), "label": s["label"]}

    ds     = CNNDataset()
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
        collate_fn=lambda b: (torch.stack([x["image"] for x in b]),
                              torch.tensor([x["label"] for x in b])),
    )

    model = ResNet18OCT(num_classes=cfg.cnn_classes, use_pretrained=False)
    ckpt  = torch.load(cfg.cnn_ckpt, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(cfg.device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  CNN"):
            preds.extend(model(imgs.to(cfg.device)).argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())

    preds, labels = np.array(preds), np.array(labels)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, target_names=ds.classes, digits=4, output_dict=True)

    print(f"  Accuracy: {acc * 100:.1f}% | F1 Macro: {f1:.4f}")
    _save_confusion_matrix(confusion_matrix(labels, preds), ds.classes,
                           "CNN ResNet18 — Confusion Matrix", f"{cfg.fig_dir}/cnn_cm.png")
    _free_mem()

    return {"model": "CNN ResNet18", "dataset": "OCT5k splits_v3 test",
            "accuracy": round(acc * 100, 2), "f1_macro": round(f1, 4),
            "classification_report": report}


# 2. MedSigLIP PRETRAINED (fara fine-tuning, zero-shot)

def eval_medsiglip_pretrained() -> dict | None:
    """
    Backbone-ul MedSigLIP pur, fara niciun fine-tuning.
    Arata performanta de baza a modelului inainte de antrenare pe datele noastre.
    Clasificare zero-shot: argmax pe similaritatea cosine cu embedding-urile text.
    """
    print("\n  EVAL: MedSigLIP pretrained (zero-shot, fara fine-tuning)")

    if not os.path.exists(cfg.med_model):
        print(f"  SKIP: {cfg.med_model} not found")
        return None

    ds, loader, proc = _make_oct5k_loader()

    backbone = AutoModel.from_pretrained(cfg.med_model, torch_dtype=torch.float32)
    backbone = backbone.to(cfg.device).eval()

    def _pool(out):
        if hasattr(out, "pooler_output"):      return out.pooler_output
        if hasattr(out, "last_hidden_state"):  return out.last_hidden_state[:, 0]
        return out

    def _enc_img(pv):
        return F.normalize(_pool(backbone.get_image_features(pixel_values=pv)), p=2, dim=-1)

    def _enc_txt(ids, mask):
        return F.normalize(_pool(backbone.get_text_features(input_ids=ids, attention_mask=mask)), p=2, dim=-1)

    # Prompturi reprezentative per clasa pt clasificare zero-shot
    # Un prompt per clasa — comparam fiecare imagine cu acestea
    CLASS_PROMPTS = {
        "AMD":    "Age-related macular degeneration with drusen deposits and retinal pigment epithelium abnormalities",
        "DME":    "Diabetic macular edema with intraretinal fluid and retinal thickening",
        "DRUSEN": "Drusen deposits beneath the retinal pigment epithelium without advanced AMD",
        "NORMAL": "Normal healthy retina with no pathological findings or fluid",
    }

    proc = AutoProcessor.from_pretrained(cfg.med_model)

    # Encodam un embedding per clasa
    class_embs = []
    for cls_name in CLASSES:
        tok  = proc.tokenizer(CLASS_PROMPTS[cls_name], padding="max_length",
                              truncation=True, max_length=64, return_tensors="pt")
        ids  = tok["input_ids"].to(cfg.device)
        mask = tok.get("attention_mask", torch.ones_like(ids)).to(cfg.device)
        with torch.no_grad():
            emb = _enc_txt(ids, mask)
        class_embs.append(emb)
    # [n_classes, dim]
    class_matrix = torch.cat(class_embs, dim=0)

    all_img, all_txt, all_lbl = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Pretrained"):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
            ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
            ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
            mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

            with autocast(cfg.device, enabled=cfg.use_amp):
                ie = _enc_img(pv)
                ea = _enc_txt(ia, ma)
                eb = _enc_txt(ib, mb)
                te = F.normalize((ea + eb) / 2, p=2, dim=-1)

            all_img.append(ie.cpu())
            all_txt.append(te.cpu())
            all_lbl.append(batch["label"])

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels  = torch.cat(all_lbl)

    retrieval = compute_retrieval_metrics(img_emb, txt_emb, labels)
    avg_r1    = (retrieval["I2T_R@1"] + retrieval["T2I_R@1"]) / 2

    # Clasificare zero-shot: fiecare imagine vs embedding-ul clasei
    sim_cls = img_emb @ class_matrix.cpu().T  # [N, n_classes]
    preds   = sim_cls.argmax(dim=1).numpy()
    acc     = accuracy_score(labels.numpy(), preds)
    f1      = f1_score(labels.numpy(), preds, average="macro")

    print(f"  Avg R@1: {avg_r1:.1f}% | Cls zero-shot: {acc * 100:.1f}% | F1: {f1:.4f}")
    _free_mem()

    return {"model": "MedSigLIP pretrained", "dataset": "OCT5k splits_v3 test",
            "accuracy": round(acc * 100, 2), "f1_macro": round(f1, 4),
            "avg_R@1": round(avg_r1, 2), **retrieval}


# 3. MedSigLIP v13 FINE-TUNED

def eval_medsiglip_v13() -> dict | None:
    print("\n  EVAL: MedSigLIP v13 fine-tuned (multi-task)")

    if not os.path.exists(cfg.med_ckpt):
        print(f"  SKIP: {cfg.med_ckpt} not found")
        return None

    ds, loader, proc = _make_oct5k_loader()

    ckpt      = torch.load(cfg.med_ckpt, map_location="cpu", weights_only=False)
    n_classes = ckpt.get("num_classes", 4)
    classes   = ckpt.get("classes", CLASSES)

    # Detectam cls_head hidden dim din checkpoint
    raw_state = ckpt.get("model", ckpt)
    cls_hidden = 256
    for key in ["classification_head.1.weight", "cls_head.1.weight"]:
        if key in raw_state:
            cls_hidden = raw_state[key].shape[0]
            break

    model = MedSigLIPMultiTask(cfg.med_model, n_classes=n_classes, cls_hidden=cls_hidden)

    # Remapam cheile vechi -> noi inainte de load
    remapped_state = _remap_checkpoint_keys(raw_state)
    model.load_state_dict(remapped_state, strict=False)
    model = model.to(cfg.device).eval()

    all_img, all_txt, all_lbl     = [], [], []
    all_sev_pred, all_sev_lbl     = [], []
    all_cls_pred                   = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  v13"):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
            ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
            ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
            mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

            with autocast(cfg.device, enabled=cfg.use_amp):
                img_emb, _, _, fused_emb, _, sev_pred, cls_logits = model(pv, ia, ma, ib, mb)

            all_img.append(img_emb.cpu())
            all_txt.append(fused_emb.cpu())
            all_lbl.append(batch["label"])
            all_sev_pred.append(sev_pred.cpu())
            all_sev_lbl.append(batch["severity"])
            all_cls_pred.append(cls_logits.argmax(1).cpu())

    _free_mem()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels  = torch.cat(all_lbl)

    retrieval     = compute_retrieval_metrics(img_emb, txt_emb, labels)
    avg_r1        = (retrieval["I2T_R@1"] + retrieval["T2I_R@1"]) / 2
    sev_pred_pct  = torch.cat(all_sev_pred) * 100
    sev_label_pct = torch.cat(all_sev_lbl)  * 100
    sev_mae       = (sev_pred_pct - sev_label_pct).abs().mean().item()
    cls_preds     = torch.cat(all_cls_pred).numpy()
    cls_labels    = labels.numpy()
    acc           = accuracy_score(cls_labels, cls_preds)
    f1            = f1_score(cls_labels, cls_preds, average="macro")
    report        = classification_report(cls_labels, cls_preds, target_names=classes, digits=4, output_dict=True)

    print(f"  Avg R@1: {avg_r1:.1f}% | Cls: {acc * 100:.1f}% | F1: {f1:.4f} | SevMAE: {sev_mae:.1f}%")

    _save_confusion_matrix(confusion_matrix(cls_labels, cls_preds), classes,
                           "MedSigLIP v13 — Confusion Matrix",
                           f"{cfg.fig_dir}/medsiglip_v13_cm.png", cmap="Greens")
    _save_severity_plots(sev_pred_pct, sev_label_pct, cls_labels, classes, sev_mae)

    return {"model": "MedSigLIP v13", "dataset": "OCT5k splits_v3 test",
            "accuracy": round(acc * 100, 2), "f1_macro": round(f1, 4),
            "avg_R@1": round(avg_r1, 2), "severity_mae": round(sev_mae, 2),
            "classification_report": report, **retrieval}


def _save_severity_plots(sev_pred, sev_label, cls_labels, classes, sev_mae):
    sp, sl = sev_pred.numpy(), sev_label.numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(sl, sp, alpha=0.3, s=10)
    plt.plot([0, 100], [0, 100], "r--", label="Perfect")
    plt.xlabel("True Severity (%)"); plt.ylabel("Predicted Severity (%)")
    plt.title(f"MedSigLIP v13 — Severity (MAE={sev_mae:.1f}%)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/medsiglip_sev_scatter.png", dpi=150); plt.close()

    plt.figure(figsize=(10, 6))
    for i, cls_name in enumerate(classes):
        mask = cls_labels == i
        if not mask.any(): continue
        mae_c = np.abs(sl[mask] - sp[mask]).mean()
        plt.scatter(sl[mask], sp[mask], alpha=0.4, s=15, label=f"{cls_name} (MAE={mae_c:.1f}%)")
    plt.plot([0, 100], [0, 100], "r--", alpha=0.5)
    plt.xlabel("True Severity (%)"); plt.ylabel("Predicted Severity (%)")
    plt.title("Severity per Disease"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/medsiglip_sev_per_class.png", dpi=150); plt.close()


# COMPARISON PLOTS

def plot_comparison(results: list) -> None:
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    acc_data = [(r["model"], r["accuracy"]) for r in results if "accuracy" in r]
    if acc_data:
        names, vals = zip(*acc_data)
        bars = axes[0].bar(names, vals, color=colors[:len(names)])
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{v}%", ha="center", fontweight="bold")
        axes[0].set(ylabel="Accuracy %", title="Classification Accuracy", ylim=(0, 105))
        axes[0].grid(alpha=0.3, axis="y")

    ret_data = [r for r in results if "I2T_R@1" in r]
    if ret_data:
        x = np.arange(3)
        w = 0.35
        for i, r in enumerate(ret_data):
            vals = [(r.get(f"I2T_R@{k}", 0) + r.get(f"T2I_R@{k}", 0)) / 2 for k in [1, 5, 10]]
            axes[1].bar(x + i * w, vals, w, label=r["model"], color=colors[i % len(colors)])
        axes[1].set_xticks(x + w / 2)
        axes[1].set_xticklabels(["R@1", "R@5", "R@10"])
        axes[1].set(ylabel="Avg Retrieval %", title="Retrieval Performance")
        axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

    caps = {r["model"]: [r.get("accuracy", 0), r.get("avg_R@1", 0),
                         max(0, 100 - r.get("severity_mae", 100))] for r in results}
    if caps:
        x = np.arange(3)
        w = 0.25
        for i, (name, vals) in enumerate(caps.items()):
            axes[2].bar(x + i * w, vals, w, label=name, color=colors[i % len(colors)])
        axes[2].set_xticks(x + w)
        axes[2].set_xticklabels(["Classification", "Retrieval", "Severity"])
        axes[2].set(ylabel="Score %", title="Model Capabilities")
        axes[2].legend(); axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{cfg.fig_dir}/comparison.png", dpi=150)
    plt.close()
    print(f"\n  Comparison plot: {cfg.fig_dir}/comparison.png")


# MAIN

def main():
    set_seed()
    print("  EVALUARE COMPARATIVA: CNN vs MedSigLIP pretrained vs MedSigLIP v13")

    results = []
    for eval_fn in [eval_cnn, eval_medsiglip_pretrained, eval_medsiglip_v13]:
        r = eval_fn()
        results.append(r)
        _free_mem()

    valid = [r for r in results if r is not None]

    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False, default=str)

    plot_comparison(valid)

    print("\n  RESULTS SUMMARY")
    for r in valid:
        print(f"\n  {r['model']} ({r['dataset']}):")
        if "accuracy"     in r: print(f"    Accuracy:     {r['accuracy']}%")
        if "f1_macro"     in r: print(f"    F1 Macro:     {r['f1_macro']}")
        if "avg_R@1"      in r: print(f"    Avg R@1:      {r['avg_R@1']}%")
        if "severity_mae" in r: print(f"    Severity MAE: {r['severity_mae']}%")
        if "I2T_R@1"      in r:
            print(f"    I2T: R@1={r['I2T_R@1']}% R@5={r['I2T_R@5']}% R@10={r['I2T_R@10']}%")
            print(f"    T2I: R@1={r['T2I_R@1']}% R@5={r['T2I_R@5']}% R@10={r['T2I_R@10']}%")

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'R@1':>8} {'F1':>8} {'SevMAE':>10}")
    print(f"  {'-' * 65}")
    for r in valid:
        acc = f"{r['accuracy']}%"      if "accuracy"     in r else "-"
        r1  = f"{r['avg_R@1']}%"      if "avg_R@1"      in r else "-"
        f1  = f"{r['f1_macro']}"       if "f1_macro"      in r else "-"
        sev = f"{r['severity_mae']}%"  if "severity_mae"  in r else "-"
        print(f"  {r['model']:<25} {acc:>10} {r1:>8} {f1:>8} {sev:>10}")

    print(f"\n  Results: {cfg.results_json}")
    print(f"  Figures: {cfg.fig_dir}/")


if __name__ == "__main__":
    main()