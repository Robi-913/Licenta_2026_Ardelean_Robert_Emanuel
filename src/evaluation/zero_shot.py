"""
Zero-Shot Evaluation — MedSigLIP v7

Disease classification — 4 setup-uri:
  [A] Single generic prompt
  [B] Prompt ensemble generic (3 variante)
  [C] Structure + Pathology (2 branch-uri cu fusion)
  [D] Structure + Pathology + Ensemble (best expected)

Biomarker detection:
  [2] Zero-shot pos/neg text (all bbox + doctor-only)
  [3] Trained heads v5 (all bbox + doctor-only)

Rulare:
    python src/evaluation/zero_shot.py
"""

import json
import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.utils.seed import set_seed


# ================================================================
# CONFIG
# ================================================================

class Config:
    model_path  = "models/medsiglip-448"
    ckpt_path   = "experiments/medsiglip_v13/ckpts/final_with_probe.pth"
    bm_ckpt_v5  = "experiments/biomarker_heads_v5/ckpts/best.pth"

    test_csv    = "data/oct5k/splits_v3/test.csv"
    split_json  = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    sev_json    = "data/OCT5k/severity_scores_v2.json"
    master_json = "data/OCT5k/metadata_v2/_master.json"

    out_json = "experiments/zero_shot_v12_results.json"

    bs      = 8
    workers = 0
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    amp     = torch.cuda.is_available()


cfg = Config()


# ================================================================
# TEXTE DISEASE — 4 setup-uri
# ================================================================

DISEASE_SINGLE = {
    "AMD":    "age-related macular degeneration retinal OCT scan",
    "DME":    "diabetic macular edema retinal OCT scan",
    "DRUSEN": "drusen deposits retinal OCT scan",
    "NORMAL": "normal healthy retina OCT scan",
}

DISEASE_ENSEMBLE = {
    "AMD": [
        "OCT scan showing age-related macular degeneration",
        "Retinal OCT with drusen deposits and macular degeneration",
        "AMD with structural changes in the macula",
    ],
    "DME": [
        "OCT scan showing diabetic macular edema",
        "Retinal OCT with intraretinal fluid from diabetic macular edema",
        "DME with retinal thickening and fluid accumulation",
    ],
    "DRUSEN": [
        "OCT scan showing drusen deposits beneath the retinal pigment epithelium",
        "Retinal OCT with early AMD drusen",
        "Small drusen deposits in the retina without advanced disease",
    ],
    "NORMAL": [
        "Normal retinal OCT with healthy layered structure",
        "OCT scan showing no pathological findings",
        "Healthy retina with normal layer thicknesses",
    ],
}

DISEASE_STRUCT = {
    "AMD":    "Retinal layer irregularities with localized thickening and thinning and altered retinal morphology",
    "DME":    "Retinal thickening with fluid related structural distortion and edema",
    "DRUSEN": "Retinal layer irregularities with localized drusen related thickening beneath the RPE",
    "NORMAL": "Preserved retinal layer organization with normal retinal morphology",
}

DISEASE_PATHO = {
    "AMD":    "Geographic atrophy and drusen deposits characteristic of age related macular degeneration",
    "DME":    "Intraretinal fluid and diabetic macular edema with retinal thickening",
    "DRUSEN": "Soft and hard drusen deposits beneath the retinal pigment epithelium without advanced AMD",
    "NORMAL": "No pathological biomarkers detected and no drusen or fluid present",
}

DISEASE_STRUCT_ENS = {
    "AMD": [
        "Retinal layer irregularities with localized thickening and thinning and altered retinal morphology",
        "Disrupted retinal architecture with RPE abnormalities and photoreceptor layer changes",
        "Macular structural changes with irregular layer boundaries and retinal thinning",
    ],
    "DME": [
        "Retinal thickening with fluid related structural distortion and edema",
        "Increased retinal thickness with cystoid spaces and disrupted layer organization",
        "Macular edema with retinal layer distortion and subretinal fluid",
    ],
    "DRUSEN": [
        "Retinal layer irregularities with localized drusen related thickening beneath the RPE",
        "RPE irregularities with dome shaped elevations and drusen deposits",
        "Subtle retinal layer changes with small focal thickenings at the RPE level",
    ],
    "NORMAL": [
        "Preserved retinal layer organization with normal retinal morphology",
        "Well defined retinal layers with uniform thickness and no structural abnormalities",
        "Normal foveal contour with intact photoreceptor layer and RPE",
    ],
}

DISEASE_PATHO_ENS = {
    "AMD": [
        "Geographic atrophy and drusen deposits characteristic of age related macular degeneration",
        "Soft drusen and retinal pigment epithelium abnormalities associated with AMD",
        "Macular degeneration with drusen accumulation and photoreceptor disruption",
    ],
    "DME": [
        "Intraretinal fluid and diabetic macular edema with retinal thickening",
        "Cystoid macular edema with fluid filled spaces in the retinal layers",
        "Diabetic retinopathy with macular edema and fluid accumulation",
    ],
    "DRUSEN": [
        "Soft and hard drusen deposits beneath the retinal pigment epithelium without advanced AMD",
        "Early AMD drusen without geographic atrophy or neovascularization",
        "Multiple drusen deposits with RPE changes and no fluid or atrophy",
    ],
    "NORMAL": [
        "No pathological biomarkers detected and no drusen or fluid present",
        "Healthy retina with no signs of drusen geographic atrophy or fluid",
        "Normal OCT with no retinal pathology identified",
    ],
}


# ================================================================
# TEXTE BIOMARKERI
# ================================================================

BIOMARKER_TEXTS = {
    "Fluid": ("There is fluid present in the retinal layers", "There is no fluid in the retinal layers"),
    "SoftDrusenPED": ("There is a soft drusen pigment epithelial detachment present", "There is no pigment epithelial detachment present"),
    "PRLayerDisruption": ("There is disruption of the photoreceptor layer", "The photoreceptor layer is intact with no disruption"),
    "GeographicAtrophy": ("There is geographic atrophy present in the retina", "There is no geographic atrophy in the retina"),
    "SoftDrusen": ("There are soft drusen deposits present", "There are no soft drusen deposits present"),
    "ReticularDrusen": ("There are reticular drusen deposits present", "There are no reticular drusen deposits present"),
    "HyperfluorescentSpots": ("There are hyperfluorescent spots present in the retina", "There are no hyperfluorescent spots in the retina"),
    "HardDrusen": ("There are hard drusen deposits present", "There are no hard drusen deposits present"),
    "ChoroidalFolds": ("There are choroidal folds present", "There are no choroidal folds present"),
}

BIOMARKER_NORMALIZE = {
    "choroidalfolds": "ChoroidalFolds", "fluid": "Fluid", "geographicatrophy": "GeographicAtrophy",
    "harddrusen": "HardDrusen", "hyperfluorescentspots": "HyperfluorescentSpots",
    "prlayerdisruption": "PRLayerDisruption", "reticulardrusen": "ReticularDrusen",
    "softdrusen": "SoftDrusen", "softdrusenped": "SoftDrusenPED",
}

BIOMARKERS_TRAINED = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]

TRAINED_TO_ZS = {
    "Fluid": "Fluid", "Geographicatrophy": "GeographicAtrophy",
    "PRlayerdisruption": "PRLayerDisruption", "SoftdrusenPED": "SoftDrusenPED",
    "Reticulardrusen": "ReticularDrusen", "Hyperfluorescentspots": "HyperfluorescentSpots",
    "Softdrusen": "SoftDrusen", "Harddrusen": "HardDrusen", "Choroidalfolds": "ChoroidalFolds",
}


# ================================================================
# MODELS
# ================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, emb_a, emb_b):
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)
        attn_a, _ = self.attn_a2b(query=a, key=b, value=b)
        attn_b, _ = self.attn_b2a(query=b, key=a, value=a)
        attn_a, attn_b = attn_a.squeeze(1), attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


def _detect_cls_head(state_dict):
    """detecteaza automat arhitectura cls_head din checkpoint"""
    # cauta primul Linear din cls_head
    for k, v in state_dict.items():
        if k == "cls_head.1.weight":
            hidden = v.shape[0]  # 256 = v13 original, 512 = probed
            return hidden
    return 256  # fallback


class MedSigLIPMultiTask(nn.Module):
    def __init__(self, model_path, n_classes=4, cls_hidden=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.backbone = get_peft_model(self.backbone, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        ))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))
        bb = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        dim = bb.config.vision_config.hidden_size
        self.sev_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1),
        )

        # cls_head — se adapteaza la checkpoint
        if cls_hidden == 512:
            # probed (linear_probe.py)
            self.cls_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(256, n_classes),
            )
        else:
            # original v13
            self.cls_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
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


class BiomarkerHeadsV5(nn.Module):
    """backbone+LoRA frozen, 9 heads trainable."""

    def __init__(self, backbone, dim, n_bm=9):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            for _ in range(n_bm)
        ])

    def encode_image(self, pixel_values):
        with torch.no_grad():
            out = self.backbone.get_image_features(pixel_values=pixel_values)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state[:, 0]
        return out  # ne-normalizat

    def forward(self, pixel_values):
        img_feat = self.encode_image(pixel_values)
        logits = torch.cat([h(img_feat) for h in self.heads], dim=-1)
        return logits


def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ================================================================
# ENCODE TEXTE
# ================================================================

def encode_single(model, proc, text, device):
    tok  = proc.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids  = tok["input_ids"].to(device)
    mask = tok.get("attention_mask", torch.ones_like(ids)).to(device)
    with torch.no_grad():
        emb = model.encode_text(ids, mask)
    return F.normalize(emb, p=2, dim=-1)


def encode_ensemble(model, proc, texts, device):
    all_emb = [encode_single(model, proc, t, device) for t in texts]
    stacked = torch.cat(all_emb, dim=0)
    return F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1)


# ================================================================
# GROUND TRUTH BIOMARKERI
# ================================================================

def build_biomarker_gt(master_json, image_paths, bbox_source_filter=None):
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_index = {m["image_path"]: m for m in metadata}
    gt = {}
    for path in image_paths:
        meta = meta_index.get(path)
        if meta is None or not meta.get("has_bounding_boxes"):
            continue
        if bbox_source_filter and meta.get("bbox_source", "doctor") != bbox_source_filter:
            continue
        present = set()
        for les in meta.get("lesions", []):
            norm = BIOMARKER_NORMALIZE.get(les.get("class", "").lower().replace(" ", "").replace("_", ""))
            if norm:
                present.add(norm)
        gt[path] = {bm: int(bm in present) for bm in BIOMARKER_TEXTS.keys()}
    return gt


def get_doctor_only_paths(master_json, image_paths):
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_index = {m["image_path"]: m for m in metadata}
    doctor_paths = set()
    for path in image_paths:
        meta = meta_index.get(path)
        if meta and meta.get("has_bounding_boxes") and meta.get("bbox_source", "doctor") == "doctor":
            doctor_paths.add(path)
    return doctor_paths


# ================================================================
# DISEASE ZERO-SHOT
# ================================================================

@torch.no_grad()
def run_disease_zs(model, loader, cls_matrix, classes, setup_name):
    all_preds, all_labels, all_confs = [], [], []
    for batch in tqdm(loader, desc=f"  {setup_name}", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.amp):
            img_emb = model.encode_image(pv)
        sim   = img_emb @ cls_matrix.T
        probs = torch.softmax(sim * 10, dim=1)
        all_preds.extend(sim.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
        all_confs.extend(probs.max(dim=1).values.cpu().numpy())
        del pv, img_emb, sim, probs
    clear_mem()
    preds, labels, confs = np.array(all_preds), np.array(all_labels), np.array(all_confs)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    return {
        "setup": setup_name, "accuracy": round(acc * 100, 2), "f1_macro": round(f1, 4),
        "mean_confidence": round(float(confs.mean() * 100), 2),
        "per_class_report": classification_report(labels, preds, target_names=classes, digits=4, output_dict=True),
    }


@torch.no_grad()
def zero_shot_disease_all(model, loader, proc, classes):
    print("\n  ===== DISEASE ZERO-SHOT — 4 setup-uri =====\n")
    results = {}

    print("  [A] Single generic prompt...")
    mat_a = torch.cat([encode_single(model, proc, DISEASE_SINGLE.get(c, c), cfg.device) for c in classes], dim=0)
    r_a = run_disease_zs(model, loader, mat_a, classes, "A_single")
    results["A_single_generic"] = r_a
    print(f"      Accuracy: {r_a['accuracy']}% | F1: {r_a['f1_macro']}")

    print("  [B] Ensemble generic (3 variante)...")
    mat_b = torch.cat([encode_ensemble(model, proc, DISEASE_ENSEMBLE.get(c, [c]), cfg.device) for c in classes], dim=0)
    r_b = run_disease_zs(model, loader, mat_b, classes, "B_ensemble")
    results["B_ensemble_generic"] = r_b
    print(f"      Accuracy: {r_b['accuracy']}% | F1: {r_b['f1_macro']}")

    print("  [C] Structure + Pathology (un text per branch)...")
    cls_vecs_c = []
    for cls in classes:
        ea = encode_single(model, proc, DISEASE_STRUCT.get(cls, cls), cfg.device)
        eb = encode_single(model, proc, DISEASE_PATHO.get(cls, cls), cfg.device)
        cls_vecs_c.append(model.fusion(ea, eb))
    mat_c = torch.cat(cls_vecs_c, dim=0)
    r_c = run_disease_zs(model, loader, mat_c, classes, "C_struct_path")
    results["C_struct_path"] = r_c
    print(f"      Accuracy: {r_c['accuracy']}% | F1: {r_c['f1_macro']}")

    print("  [D] Structure + Pathology + Ensemble (3 variante per branch)...")
    cls_vecs_d = []
    for cls in classes:
        ea = encode_ensemble(model, proc, DISEASE_STRUCT_ENS.get(cls, [cls]), cfg.device)
        eb = encode_ensemble(model, proc, DISEASE_PATHO_ENS.get(cls, [cls]), cfg.device)
        cls_vecs_d.append(model.fusion(ea, eb))
    mat_d = torch.cat(cls_vecs_d, dim=0)
    r_d = run_disease_zs(model, loader, mat_d, classes, "D_struct_path_ens")
    results["D_struct_path_ensemble"] = r_d
    print(f"      Accuracy: {r_d['accuracy']}% | F1: {r_d['f1_macro']}")

    print(f"\n  {'Setup':<35} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*57}")
    for name, r in results.items():
        print(f"  {name:<35} {r['accuracy']:>9}% {r['f1_macro']:>10.4f}")
    return results


# ================================================================
# BIOMARKER EVALUATION HELPERS
# ================================================================

@torch.no_grad()
def get_bbox_embeddings(model, bbox_loader, dataset_bbox):
    all_img_embs, all_paths = [], []
    for batch_idx, batch in enumerate(tqdm(bbox_loader, desc="  Encoding", leave=False)):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.amp):
            img_emb = model.encode_image(pv)
        all_img_embs.append(img_emb.cpu())
        start = batch_idx * cfg.bs
        end   = min(start + cfg.bs, len(dataset_bbox))
        for i in range(end - start):
            all_paths.append(dataset_bbox.df.iloc[start + i]["image_path"])
        del pv, img_emb
    clear_mem()
    return torch.cat(all_img_embs), all_paths


def _eval_biomarker_preds(predicted, all_paths, gt, bm_names, label, display_names=None):
    if display_names is None:
        display_names = bm_names

    results_per_bm = {}
    all_gt_flat, all_pred_flat = [], []

    print(f"\n  {'Biomarker':<25} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'GT+':>5}")
    print(f"  {'-'*60}")

    for bm_idx, bm_name in enumerate(bm_names):
        if bm_name is None:
            continue
        display = display_names[bm_idx]

        gt_labels, pred_labels = [], []
        for img_idx, path in enumerate(all_paths):
            if path not in gt or bm_name not in gt[path]:
                continue
            gt_labels.append(gt[path][bm_name])
            pred_labels.append(int(predicted[img_idx, bm_idx]))
        if not gt_labels:
            continue

        gt_arr, pred_arr = np.array(gt_labels), np.array(pred_labels)
        n_pos = int(gt_arr.sum())
        if n_pos == 0:
            continue

        acc  = accuracy_score(gt_arr, pred_arr)
        f1   = f1_score(gt_arr, pred_arr, zero_division=0)
        prec = precision_recall(gt_arr, pred_arr, "precision")
        rec  = precision_recall(gt_arr, pred_arr, "recall")

        print(f"  {display:<25} {acc*100:>5.1f}% {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {n_pos:>5}")
        results_per_bm[display] = {
            "accuracy": round(acc * 100, 2), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "n_positive": n_pos,
        }
        all_gt_flat.extend(gt_labels)
        all_pred_flat.extend(pred_labels)

    macro_f1 = overall_acc = overall_f1 = None
    if all_gt_flat:
        overall_acc = accuracy_score(all_gt_flat, all_pred_flat)
        overall_f1  = f1_score(all_gt_flat, all_pred_flat, zero_division=0)
        f1s = [v["f1"] for v in results_per_bm.values() if v["n_positive"] > 0]
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        print(f"\n  [{label}] Macro F1={macro_f1:.4f} | Overall F1={overall_f1:.4f} | Acc={overall_acc*100:.2f}%")

    return {
        "per_biomarker":    results_per_bm,
        "overall_accuracy": round(overall_acc * 100, 2) if overall_acc else None,
        "overall_f1":       round(overall_f1, 4) if overall_f1 else None,
        "macro_f1":         round(macro_f1, 4) if macro_f1 is not None else None,
    }


def precision_recall(gt_arr, pred_arr, metric):
    from sklearn.metrics import precision_score, recall_score
    if metric == "precision":
        return precision_score(gt_arr, pred_arr, zero_division=0)
    else:
        return recall_score(gt_arr, pred_arr, zero_division=0)


# ================================================================
# BIOMARKER ZERO-SHOT
# ================================================================

@torch.no_grad()
def zero_shot_biomarkers(model, bbox_loader, dataset_bbox, gt, proc, label="Zero-shot"):
    print(f"\n  [ZS] {label} ({sum(1 for p in dataset_bbox.df['image_path'] if p in gt)} imagini cu GT)...")

    bm_names = list(BIOMARKER_TEXTS.keys())
    pos_matrix = torch.cat([
        encode_single(model, proc, BIOMARKER_TEXTS[bm][0], cfg.device) for bm in bm_names
    ], dim=0).to(cfg.device)
    neg_matrix = torch.cat([
        encode_single(model, proc, BIOMARKER_TEXTS[bm][1], cfg.device) for bm in bm_names
    ], dim=0).to(cfg.device)

    img_embs, all_paths = get_bbox_embeddings(model, bbox_loader, dataset_bbox)
    img_embs = img_embs.to(cfg.device)
    predicted = (img_embs @ pos_matrix.T > img_embs @ neg_matrix.T).cpu().numpy()

    return _eval_biomarker_preds(predicted, all_paths, gt, bm_names, label)


# ================================================================
# BIOMARKER TRAINED HEADS V5
# ================================================================

@torch.no_grad()
def trained_biomarker_v5(bm_model, bbox_loader, dataset_bbox, gt, thresholds, label="v5 trained"):
    print(f"\n  [TH] {label} ({sum(1 for p in dataset_bbox.df['image_path'] if p in gt)} imagini cu GT)...")
    print(f"      Thresholds: {[round(t, 2) for t in thresholds]}")

    all_probs, all_paths = [], []

    for batch_idx, batch in enumerate(tqdm(bbox_loader, desc=f"  {label}", leave=False)):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.amp):
            logits = bm_model(pv)
            probs  = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        start = batch_idx * cfg.bs
        end   = min(start + cfg.bs, len(dataset_bbox))
        for i in range(end - start):
            all_paths.append(dataset_bbox.df.iloc[start + i]["image_path"])
        del pv, logits, probs

    clear_mem()
    probs = torch.cat(all_probs).numpy()

    predicted = np.zeros_like(probs)
    for i in range(len(BIOMARKERS_TRAINED)):
        predicted[:, i] = (probs[:, i] > thresholds[i]).astype(float)

    bm_names_mapped = [TRAINED_TO_ZS.get(bm) for bm in BIOMARKERS_TRAINED]
    return _eval_biomarker_preds(predicted, all_paths, gt, bm_names_mapped, label, BIOMARKERS_TRAINED)


# ================================================================
# MAIN
# ================================================================

def main():
    set_seed()

    print("=" * 70)
    print("  ZERO-SHOT EVALUATION — MedSigLIP v7")
    print("  Disease: 4 setup-uri (A/B/C/D)")
    print("  Biomarker: zero-shot + trained heads v5")
    print("  Eval pe: ALL bbox + DOCTOR-ONLY bbox (comparatie corecta)")
    print("=" * 70)

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    # --- MedSigLIP model ---
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    nc = 4
    classes = ["AMD", "DME", "DRUSEN", "NORMAL"]

    # auto-detect cls_head din checkpoint
    cls_hidden = _detect_cls_head(state_dict)
    print(f"  cls_head detected: hidden={cls_hidden} ({'probed' if cls_hidden == 512 else 'original'})")

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc, cls_hidden=cls_hidden)
    model.load_state_dict(state_dict)
    model = model.to(cfg.device)
    model.eval()
    print(f"  MedSigLIP: {cfg.ckpt_path}")

    # --- Biomarker heads v5 (LoRA backbone din bm_ckpt) ---
    bm_model = None
    bm_thresholds = [0.5] * len(BIOMARKERS_TRAINED)

    if os.path.exists(cfg.bm_ckpt_v5):
        bm_ckpt = torch.load(cfg.bm_ckpt_v5, map_location="cpu", weights_only=False)

        backbone_v5 = AutoModel.from_pretrained(cfg.model_path, torch_dtype=torch.float32)
        backbone_v5 = get_peft_model(backbone_v5, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        ))

        bb_state = {
            k.replace("backbone.", ""): v
            for k, v in bm_ckpt["model"].items()
            if k.startswith("backbone.")
        }
        backbone_v5.load_state_dict(bb_state, strict=True)

        bb = backbone_v5.base_model.model if hasattr(backbone_v5, "base_model") else backbone_v5
        dim = bb.config.vision_config.hidden_size

        bm_model = BiomarkerHeadsV5(backbone_v5, dim, n_bm=len(BIOMARKERS_TRAINED))
        bm_model.load_state_dict(bm_ckpt["model"])
        bm_model = bm_model.to(cfg.device)
        bm_model.eval()

        bm_thresholds = [float(t) for t in bm_ckpt.get("thresholds", [0.5] * len(BIOMARKERS_TRAINED))]
        print(f"  Biomarker v5 (LoRA): {cfg.bm_ckpt_v5}")
        print(f"  Thresholds: {[round(t, 2) for t in bm_thresholds]}")
    else:
        print(f"  Biomarker v5 not found — skipping")

    # --- loaders ---
    ds_full = OCT5kDataset(
        split_csv=cfg.test_csv, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=proc, mode="eval",
    )
    loader_full = DataLoader(
        ds_full, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    # --- ALL bbox loader ---
    test_df      = pd.read_csv(cfg.test_csv)
    bbox_df_all  = test_df[test_df["has_bbox"] == True].reset_index(drop=True)
    bbox_csv_all = cfg.test_csv.replace("test.csv", "_tmp_bbox_all.csv")
    bbox_df_all.to_csv(bbox_csv_all, index=False)

    ds_bbox_all = OCT5kDataset(
        split_csv=bbox_csv_all, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=proc, mode="eval",
    )
    loader_bbox_all = DataLoader(
        ds_bbox_all, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    # --- DOCTOR-ONLY bbox loader ---
    doctor_paths = get_doctor_only_paths(cfg.master_json, bbox_df_all["image_path"].tolist())
    bbox_df_doc  = bbox_df_all[bbox_df_all["image_path"].isin(doctor_paths)].reset_index(drop=True)
    bbox_csv_doc = cfg.test_csv.replace("test.csv", "_tmp_bbox_doctor.csv")
    bbox_df_doc.to_csv(bbox_csv_doc, index=False)

    ds_bbox_doc = OCT5kDataset(
        split_csv=bbox_csv_doc, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=proc, mode="eval",
    )
    loader_bbox_doc = DataLoader(
        ds_bbox_doc, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    # ground truth
    gt_all = build_biomarker_gt(cfg.master_json, ds_bbox_all.df["image_path"].tolist())
    gt_doc = build_biomarker_gt(cfg.master_json, ds_bbox_doc.df["image_path"].tolist(), bbox_source_filter="doctor")

    print(f"\n  Test: {len(ds_full)}")
    print(f"  Bbox ALL (doctor+yolo): {len(ds_bbox_all)} | GT entries: {len(gt_all)}")
    print(f"  Bbox DOCTOR-ONLY:       {len(ds_bbox_doc)} | GT entries: {len(gt_doc)}")
    print()

    # ---- [1] Disease zero-shot ----
    disease_results = zero_shot_disease_all(model, loader_full, proc, classes)

    # ---- [2] Biomarker — ALL bbox ----
    print("\n" + "=" * 70)
    print("  BIOMARKER EVAL — ALL BBOX (doctor + yolo)")
    print("=" * 70)
    zs_bm_all = zero_shot_biomarkers(model, loader_bbox_all, ds_bbox_all, gt_all, proc, "ZS all bbox")
    trained_all = None
    if bm_model:
        trained_all = trained_biomarker_v5(bm_model, loader_bbox_all, ds_bbox_all, gt_all, bm_thresholds, "v5 all bbox")

    # ---- [3] Biomarker — DOCTOR-ONLY ----
    print("\n" + "=" * 70)
    print("  BIOMARKER EVAL — DOCTOR-ONLY (comparatie corecta cu v3)")
    print("=" * 70)
    zs_bm_doc = zero_shot_biomarkers(model, loader_bbox_doc, ds_bbox_doc, gt_doc, proc, "ZS doctor-only")
    trained_doc = None
    if bm_model:
        trained_doc = trained_biomarker_v5(bm_model, loader_bbox_doc, ds_bbox_doc, gt_doc, bm_thresholds, "v5 doctor-only")

    # cleanup
    for f in [bbox_csv_all, bbox_csv_doc]:
        if os.path.exists(f):
            os.remove(f)

    # save
    results = {
        "model":                 "MedSigLIP_v7",
        "disease_zero_shot":     disease_results,
        "biomarker_all_bbox": {
            "zero_shot":  zs_bm_all,
            "trained_v5": trained_all,
        },
        "biomarker_doctor_only": {
            "zero_shot":  zs_bm_doc,
            "trained_v5": trained_doc,
        },
    }
    os.makedirs(os.path.dirname(cfg.out_json), exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY FINAL")
    print(f"{'=' * 70}")

    print(f"\n  Disease Zero-Shot:")
    for name, r in disease_results.items():
        print(f"    {name:<35} Acc={r['accuracy']}% | F1={r['f1_macro']}")

    print(f"\n  Biomarker — ALL bbox ({len(ds_bbox_all)} img):")
    if zs_bm_all.get("macro_f1"):
        print(f"    Zero-shot:  Macro F1={zs_bm_all['macro_f1']}")
    if trained_all and trained_all.get("macro_f1"):
        print(f"    Trained v5: Macro F1={trained_all['macro_f1']}")

    print(f"\n  Biomarker — DOCTOR-ONLY ({len(ds_bbox_doc)} img):")
    if zs_bm_doc.get("macro_f1"):
        print(f"    Zero-shot:  Macro F1={zs_bm_doc['macro_f1']}")
    if trained_doc and trained_doc.get("macro_f1"):
        print(f"    Trained v5: Macro F1={trained_doc['macro_f1']}")

    print(f"\n  Results: {cfg.out_json}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()