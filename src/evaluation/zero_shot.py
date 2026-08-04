"""
Disease classification — 4 setup-uri:
  [A] Single generic prompt
  [B] Prompt ensemble generic (3 variante)
  [C] Structure + Pathology (2 branch-uri cu fusion)
  [D] Structure + Pathology + Ensemble (best expected)

Biomarker detection:
  [2] Zero-shot pos/neg text (all bbox + doctor-only)
  [3] Trained heads v5 (all bbox + doctor-only)
"""

import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.model.medsiglip import BiomarkerHeadsV5, MedSigLIPMultiTask
from src.utils.seed import set_seed


# CONFIG

class Config:
    model_path = "models/medsiglip-448"
    ckpt_path = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"
    bm_ckpt_v5 = "experiments/biomarker_heads_v5/ckpts/best.pth"

    test_csv = "data/oct5k/splits_v3/test.csv"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    sev_json = "data/OCT5k/severity_scores_v2.json"
    master_json = "data/OCT5k/metadata_v2/_master.json"

    out_json = "experiments/zero_shot_v15_results.json"

    batch_size = 8
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()


cfg = Config()

CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]

# DISEASE PROMPTS — 4 setup-uri

DISEASE_SINGLE = {
    "AMD": "age-related macular degeneration retinal OCT scan",
    "DME": "diabetic macular edema retinal OCT scan",
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
    "AMD": "Retinal layer irregularities with localized thickening and thinning and altered retinal morphology",
    "DME": "Retinal thickening with fluid related structural distortion and edema",
    "DRUSEN": "Retinal layer irregularities with localized drusen related thickening beneath the RPE",
    "NORMAL": "Preserved retinal layer organization with normal retinal morphology",
}

DISEASE_PATHO = {
    "AMD": "Geographic atrophy and drusen deposits characteristic of age related macular degeneration",
    "DME": "Intraretinal fluid and diabetic macular edema with retinal thickening",
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

# BIOMARKER PROMPTS

# Fiecare biomarker are un prompt pozitiv si unul negativ
# Clasificam prin compararea similaritatii cu pozitivul vs negativul
BIOMARKER_TEXTS = {
    "Fluid": ("There is fluid present in the retinal layers", "There is no fluid in the retinal layers"),
    "SoftDrusenPED": ("There is a soft drusen pigment epithelial detachment present",
                      "There is no pigment epithelial detachment present"),
    "PRLayerDisruption": ("There is disruption of the photoreceptor layer",
                          "The photoreceptor layer is intact with no disruption"),
    "GeographicAtrophy": ("There is geographic atrophy present in the retina",
                          "There is no geographic atrophy in the retina"),
    "SoftDrusen": ("There are soft drusen deposits present", "There are no soft drusen deposits present"),
    "ReticularDrusen": ("There are reticular drusen deposits present",
                        "There are no reticular drusen deposits present"),
    "HyperfluorescentSpots": ("There are hyperfluorescent spots present in the retina",
                              "There are no hyperfluorescent spots in the retina"),
    "HardDrusen": ("There are hard drusen deposits present", "There are no hard drusen deposits present"),
    "ChoroidalFolds": ("There are choroidal folds present", "There are no choroidal folds present"),
}

# Normalizare pentru matching robust din metadata (case, spatii, underscore)
_BM_NORMALIZE = {
    "choroidalfolds": "ChoroidalFolds", "fluid": "Fluid", "geographicatrophy": "GeographicAtrophy",
    "harddrusen": "HardDrusen", "hyperfluorescentspots": "HyperfluorescentSpots",
    "prlayerdisruption": "PRLayerDisruption", "reticulardrusen": "ReticularDrusen",
    "softdrusen": "SoftDrusen", "softdrusenped": "SoftDrusenPED",
}

# Biomarkerii asa cum sunt salvati in checkpointul v5 (ordinea conteaza pt head index)
BIOMARKERS_TRAINED = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]

# Mapping intre numele din checkpointul v5 si cheile din BIOMARKER_TEXTS
_TRAINED_TO_ZS = {
    "Fluid": "Fluid", "Geographicatrophy": "GeographicAtrophy",
    "PRlayerdisruption": "PRLayerDisruption", "SoftdrusenPED": "SoftDrusenPED",
    "Reticulardrusen": "ReticularDrusen", "Hyperfluorescentspots": "HyperfluorescentSpots",
    "Softdrusen": "SoftDrusen", "Harddrusen": "HardDrusen", "Choroidalfolds": "ChoroidalFolds",
}


# MODEL LOADING

def _detect_cls_head_hidden(state_dict: dict) -> int:
    """
    Detecteaza automat daca cls_head din checkpoint e cel original (256)
    sau cel din linear_probe (512) — uitandu-ne la dimensiunea primului Linear.
    """
    w = state_dict.get("classification_head.1.weight")
    if w is not None:
        return w.shape[0]
    # fallback pt chei vechi (inainte de redenumire)
    w = state_dict.get("cls_head.1.weight")
    return w.shape[0] if w is not None else 256


def load_multitask_model(ckpt_path: str, n_classes: int = 4) -> MedSigLIPMultiTask:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    cls_hidden = _detect_cls_head_hidden(state_dict)
    print(f"  cls_head detected: hidden={cls_hidden} ({'probed' if cls_hidden == 512 else 'original v13'})")

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=n_classes, cls_hidden=cls_hidden)
    remapped = {
        k.replace("sev_head.", "severity_head.")
        .replace("cls_head.", "classification_head.")
        .replace("fusion.attn_a2b.", "fusion.attn_a_to_b.")
        .replace("fusion.attn_b2a.", "fusion.attn_b_to_a."): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(remapped, strict=False)
    model = model.to(cfg.device).eval()
    print(f"  MedSigLIP incarcat: {ckpt_path}")
    return model


def load_biomarker_model(bm_ckpt_path: str) -> tuple[BiomarkerHeadsV5, list[float]] | tuple[None, None]:
    """
    Incarca BiomarkerHeadsV5 din checkpointul v5.
    Returneaza (model, thresholds) sau (None, None) daca fisierul nu exista.
    """
    if not os.path.exists(bm_ckpt_path):
        print(f"  Biomarker v5 not found — skipping: {bm_ckpt_path}")
        return None, None

    bm_ckpt = torch.load(bm_ckpt_path, map_location="cpu", weights_only=False)

    # Cream backbone-ul cu aceiasi parametri ca v13, incarcam doar cheile backbone.*
    base_model = MedSigLIPMultiTask(cfg.model_path)
    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in bm_ckpt["model"].items()
        if k.startswith("backbone.")
    }
    base_model.backbone.load_state_dict(backbone_state, strict=False)

    bm_model = BiomarkerHeadsV5(backbone=base_model, n_biomarkers=len(BIOMARKERS_TRAINED))
    bm_model.load_state_dict(bm_ckpt["model"], strict=False)
    bm_model = bm_model.to(cfg.device).eval()

    thresholds = [float(t) for t in bm_ckpt.get("thresholds", [0.5] * len(BIOMARKERS_TRAINED))]
    print(f"  Biomarker v5 incarcat: {bm_ckpt_path}")
    print(f"  Thresholds: {[round(t, 2) for t in thresholds]}")
    return bm_model, thresholds


# TEXT ENCODING

@torch.no_grad()
def encode_single(model: MedSigLIPMultiTask, processor: AutoProcessor, text: str) -> torch.Tensor:
    """Encodeaza un singur text si returneaza embedding L2-normalizat [1, dim]."""
    tok = processor.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids = tok["input_ids"].to(cfg.device)
    mask = tok.get("attention_mask", torch.ones_like(ids)).to(cfg.device)
    return F.normalize(model.encode_text(ids, mask), p=2, dim=-1)


@torch.no_grad()
def encode_ensemble(model: MedSigLIPMultiTask, processor: AutoProcessor, texts: list[str]) -> torch.Tensor:
    """
    Encodeaza mai multe texte si returneaza media lor L2-normalizata [1, dim].
    Prompt ensemble reduce sensibilitatea la formularea exacta a textului.
    """
    embs = torch.cat([encode_single(model, processor, t) for t in texts], dim=0)
    return F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=-1)


# DISEASE ZERO-SHOT

@torch.no_grad()
def _run_disease_setup(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        class_matrix: torch.Tensor,
        setup_name: str,
) -> dict:
    """
    Ruleaza un setup de clasificare zero-shot.
    class_matrix: [n_classes, dim] — un embedding per clasa.
    Clasificam prin argmax pe similaritatea cosine * 10 (temperatura fixa).
    """
    all_preds, all_labels, all_confs = [], [], []

    for batch in tqdm(loader, desc=f"  {setup_name}", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.use_amp):
            img_emb = model.encode_image(pv)
            img_emb = F.normalize(img_emb, p=2, dim=-1)
        sim = img_emb @ class_matrix.T
        probs = torch.softmax(sim * 10, dim=1)  # temperatura 10 pt distributie mai ascutita
        all_preds.extend(sim.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch["label"].numpy())
        all_confs.extend(probs.max(dim=1).values.cpu().numpy())

    _free_mem()
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    confs = np.array(all_confs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {
        "setup": setup_name,
        "accuracy": round(acc * 100, 2),
        "f1_macro": round(f1, 4),
        "mean_confidence": round(float(confs.mean() * 100), 2),
        "per_class_report": classification_report(labels, preds, target_names=CLASSES, digits=4, output_dict=True),
    }


@torch.no_grad()
def evaluate_disease_zero_shot(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        processor: AutoProcessor,
) -> dict:
    """Ruleaza cele 4 setup-uri de clasificare zero-shot si afiseaza comparatia."""
    print("\n  DISEASE ZERO-SHOT — 4 setup-uri\n")
    results = {}

    print("  [A] Single generic prompt...")
    mat_a = torch.cat([encode_single(model, processor, DISEASE_SINGLE[c]) for c in CLASSES], dim=0)
    results["A_single_generic"] = _run_disease_setup(model, loader, mat_a, "A_single")
    print(f"      Accuracy: {results['A_single_generic']['accuracy']}% | F1: {results['A_single_generic']['f1_macro']}")

    print("  [B] Ensemble generic (3 variante)...")
    mat_b = torch.cat([encode_ensemble(model, processor, DISEASE_ENSEMBLE[c]) for c in CLASSES], dim=0)
    results["B_ensemble_generic"] = _run_disease_setup(model, loader, mat_b, "B_ensemble")
    print(
        f"      Accuracy: {results['B_ensemble_generic']['accuracy']}% | F1: {results['B_ensemble_generic']['f1_macro']}")

    print("  [C] Structure + Pathology (un text per branch)...")
    mat_c = torch.cat([
        model.fusion(encode_single(model, processor, DISEASE_STRUCT[c]),
                     encode_single(model, processor, DISEASE_PATHO[c]))
        for c in CLASSES
    ], dim=0)
    results["C_struct_path"] = _run_disease_setup(model, loader, mat_c, "C_struct_path")
    print(f"      Accuracy: {results['C_struct_path']['accuracy']}% | F1: {results['C_struct_path']['f1_macro']}")

    print("  [D] Structure + Pathology + Ensemble (3 variante per branch)...")
    mat_d = torch.cat([
        model.fusion(encode_ensemble(model, processor, DISEASE_STRUCT_ENS[c]),
                     encode_ensemble(model, processor, DISEASE_PATHO_ENS[c]))
        for c in CLASSES
    ], dim=0)
    results["D_struct_path_ensemble"] = _run_disease_setup(model, loader, mat_d, "D_struct_path_ens")
    print(
        f"      Accuracy: {results['D_struct_path_ensemble']['accuracy']}% | F1: {results['D_struct_path_ensemble']['f1_macro']}")

    # Tabel comparativ
    print(f"\n  {'Setup':<35} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-' * 57}")
    for name, r in results.items():
        print(f"  {name:<35} {r['accuracy']:>9}% {r['f1_macro']:>10.4f}")

    return results


# BIOMARKER GROUND TRUTH

def build_biomarker_gt(
        master_json: str,
        image_paths: list[str],
        doctor_only: bool = False,
) -> dict:
    """
    Construieste ground truth binar per biomarker din metadata.
    doctor_only=True => include doar imaginile cu bbox_source == 'doctor'.
    """
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_index = {m["image_path"]: m for m in metadata}

    gt = {}
    for path in image_paths:
        meta = meta_index.get(path)
        if meta is None or not meta.get("has_bounding_boxes"):
            continue
        if doctor_only and meta.get("bbox_source", "doctor") != "doctor":
            continue

        present = {
                      _BM_NORMALIZE.get(les.get("class", "").lower().replace(" ", "").replace("_", ""))
                      for les in meta.get("lesions", [])
                  } - {None}

        gt[path] = {bm: int(bm in present) for bm in BIOMARKER_TEXTS}

    return gt


def get_doctor_only_paths(master_json: str, image_paths: list[str]) -> set[str]:
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_index = {m["image_path"]: m for m in metadata}
    return {
        p for p in image_paths
        if meta_index.get(p, {}).get("has_bounding_boxes")
           and meta_index.get(p, {}).get("bbox_source", "doctor") == "doctor"
    }


# BIOMARKER EVALUATION — helper comun pt ZS si trained

def _eval_biomarker_predictions(
        predicted: np.ndarray,
        all_paths: list[str],
        gt: dict,
        bm_names: list[str | None],
        label: str,
        display_names: list[str] | None = None,
) -> dict:
    """
    Calculeaza F1/Acc/Precision/Recall per biomarker si macro F1.
    bm_names: cheile din BIOMARKER_TEXTS pt fiecare coloana din predicted.
    display_names: cum apar in print (poate fi diferit de bm_names).
    """
    if display_names is None:
        display_names = bm_names

    results_per_bm = {}
    all_gt_flat, all_pred_flat = [], []

    print(f"\n  {'Biomarker':<25} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'GT+':>5}")
    print(f"  {'-' * 60}")

    for col_idx, bm_name in enumerate(bm_names):
        if bm_name is None:
            continue

        gt_labels, pred_labels = zip(*[
            (gt[path][bm_name], int(predicted[img_idx, col_idx]))
            for img_idx, path in enumerate(all_paths)
            if path in gt and bm_name in gt[path]
        ]) if any(path in gt and bm_name in gt[path] for path in all_paths) else ([], [])

        if not gt_labels or int(sum(gt_labels)) == 0:
            continue

        gt_arr, pred_arr = np.array(gt_labels), np.array(pred_labels)
        n_pos = int(gt_arr.sum())
        acc = accuracy_score(gt_arr, pred_arr)
        f1 = f1_score(gt_arr, pred_arr, zero_division=0)
        prec = precision_score(gt_arr, pred_arr, zero_division=0)
        rec = recall_score(gt_arr, pred_arr, zero_division=0)

        display = display_names[col_idx]
        print(f"  {display:<25} {acc * 100:>5.1f}% {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {n_pos:>5}")

        results_per_bm[display] = {
            "accuracy": round(acc * 100, 2), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "n_positive": n_pos,
        }
        all_gt_flat.extend(gt_labels)
        all_pred_flat.extend(pred_labels)

    macro_f1 = overall_acc = overall_f1 = None
    if all_gt_flat:
        overall_acc = accuracy_score(all_gt_flat, all_pred_flat)
        overall_f1 = f1_score(all_gt_flat, all_pred_flat, zero_division=0)
        f1s = [v["f1"] for v in results_per_bm.values() if v["n_positive"] > 0]
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        print(f"\n  [{label}] Macro F1={macro_f1:.4f} | Overall F1={overall_f1:.4f} | Acc={overall_acc * 100:.2f}%")

    return {
        "per_biomarker": results_per_bm,
        "overall_accuracy": round(overall_acc * 100, 2) if overall_acc is not None else None,
        "overall_f1": round(overall_f1, 4) if overall_f1 is not None else None,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
    }


@torch.no_grad()
def _collect_image_embeddings(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        dataset: OCT5kDataset,
) -> tuple[torch.Tensor, list[str]]:
    """Ruleaza encoder vizual peste tot loader-ul si returneaza (embeddings, paths)."""
    all_embs, all_paths = [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc="  Encoding", leave=False)):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.use_amp):
            emb = model.encode_image(pv)
        all_embs.append(emb.cpu().float())
        start = batch_idx * cfg.batch_size
        for i in range(min(cfg.batch_size, len(dataset) - start)):
            all_paths.append(dataset.df.iloc[start + i]["image_path"])
    _free_mem()
    return torch.cat(all_embs), all_paths


# BIOMARKER ZERO-SHOT

@torch.no_grad()
def evaluate_biomarkers_zero_shot(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        dataset: OCT5kDataset,
        gt: dict,
        processor: AutoProcessor,
        label: str = "Zero-shot",
) -> dict:
    """
    Clasificare zero-shot per biomarker prin compararea similaritatii
    cu promptul pozitiv vs promptul negativ.
    Daca sim(img, pos) > sim(img, neg) => biomarkerul e prezent.
    """
    print(f"\n  [ZS] {label} ({sum(1 for p in dataset.df['image_path'] if p in gt)} imagini cu GT)...")

    bm_names = list(BIOMARKER_TEXTS.keys())
    pos_matrix = torch.cat([encode_single(model, processor, BIOMARKER_TEXTS[bm][0]) for bm in bm_names], dim=0)
    neg_matrix = torch.cat([encode_single(model, processor, BIOMARKER_TEXTS[bm][1]) for bm in bm_names], dim=0)

    pos_matrix = pos_matrix.float()
    neg_matrix = neg_matrix.float()

    img_embs, all_paths = _collect_image_embeddings(model, loader, dataset)
    img_embs = img_embs.to(cfg.device)

    # Prezis pozitiv daca similaritatea cu promptul pozitiv e mai mare decat cu cel negativ
    predicted = (img_embs @ pos_matrix.T > img_embs @ neg_matrix.T).cpu().numpy()

    return _eval_biomarker_predictions(predicted, all_paths, gt, bm_names, label)


# BIOMARKER TRAINED HEADS V5

@torch.no_grad()
def evaluate_biomarkers_trained(
        bm_model: BiomarkerHeadsV5,
        loader: DataLoader,
        dataset: OCT5kDataset,
        gt: dict,
        thresholds: list[float],
        label: str = "v5 trained",
) -> dict:
    """Evalueaza head-urile antrenate cu thresholds optimizate per biomarker."""
    print(f"\n  [TH] {label} ({sum(1 for p in dataset.df['image_path'] if p in gt)} imagini cu GT)...")
    print(f"       Thresholds: {[round(t, 2) for t in thresholds]}")

    all_probs, all_paths = [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"  {label}", leave=False)):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        with autocast(cfg.device, enabled=cfg.use_amp):
            probs = torch.sigmoid(bm_model(pv))
        all_probs.append(probs.cpu())
        start = batch_idx * cfg.batch_size
        for i in range(min(cfg.batch_size, len(dataset) - start)):
            all_paths.append(dataset.df.iloc[start + i]["image_path"])

    _free_mem()
    probs_np = torch.cat(all_probs).numpy()
    predicted = np.stack([
        (probs_np[:, i] > thresholds[i]).astype(float)
        for i in range(len(BIOMARKERS_TRAINED))
    ], axis=1)

    bm_names_mapped = [_TRAINED_TO_ZS.get(bm) for bm in BIOMARKERS_TRAINED]
    return _eval_biomarker_predictions(predicted, all_paths, gt, bm_names_mapped, label, BIOMARKERS_TRAINED)


# DATA LOADING HELPERS

def _make_bbox_loader(base_csv: str, image_paths: list[str], processor: AutoProcessor, tmp_suffix: str) -> tuple:
    """Construieste un loader filtrat pe imaginile din image_paths."""
    test_df = pd.read_csv(base_csv)
    filtered_df = test_df[test_df["image_path"].isin(set(image_paths))].reset_index(drop=True)
    tmp_csv = base_csv.replace("test.csv", f"_tmp_{tmp_suffix}.csv")
    filtered_df.to_csv(tmp_csv, index=False)

    ds = OCT5kDataset(
        split_csv=tmp_csv, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=processor, mode="eval",
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )
    return ds, loader, tmp_csv


def _free_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# MAIN

def main():
    set_seed()

    print("  ZERO-SHOT EVALUATION — MedSigLIP v13")
    print("  Disease: 4 setup-uri (A/B/C/D)")
    print("  Biomarker: zero-shot + trained heads v5")
    print("  Eval pe: ALL bbox + DOCTOR-ONLY bbox")

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    model = load_multitask_model(cfg.ckpt_path)
    bm_model, bm_thrs = load_biomarker_model(cfg.bm_ckpt_v5)

    # Loader principal — tot test split pt disease evaluation
    ds_full = OCT5kDataset(
        split_csv=cfg.test_csv, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=processor, mode="eval",
    )
    loader_full = DataLoader(
        ds_full, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    # Loadere separate pt biomarkeri: ALL bbox si DOCTOR-ONLY
    test_df = pd.read_csv(cfg.test_csv)
    bbox_paths_all = test_df[test_df["has_bbox"] == True]["image_path"].tolist()
    bbox_paths_doc = list(get_doctor_only_paths(cfg.master_json, bbox_paths_all))

    ds_all, loader_all, tmp_all = _make_bbox_loader(cfg.test_csv, bbox_paths_all, processor, "bbox_all")
    ds_doc, loader_doc, tmp_doc = _make_bbox_loader(cfg.test_csv, bbox_paths_doc, processor, "bbox_doc")

    gt_all = build_biomarker_gt(cfg.master_json, bbox_paths_all, doctor_only=False)
    gt_doc = build_biomarker_gt(cfg.master_json, bbox_paths_doc, doctor_only=True)

    print(f"\n  Test: {len(ds_full)}")
    print(f"  Bbox ALL (doctor+yolo): {len(ds_all)} | GT entries: {len(gt_all)}")
    print(f"  Bbox DOCTOR-ONLY:       {len(ds_doc)} | GT entries: {len(gt_doc)}")

    # Disease zero-shot — 4 setup-uri
    disease_results = evaluate_disease_zero_shot(model, loader_full, processor)

    # Biomarker zero-shot + trained — ALL bbox
    print("\n  BIOMARKER EVAL — ALL BBOX (doctor + yolo)")
    zs_all = evaluate_biomarkers_zero_shot(model, loader_all, ds_all, gt_all, processor, "ZS all bbox")
    trained_all = evaluate_biomarkers_trained(bm_model, loader_all, ds_all, gt_all, bm_thrs,
                                              "v5 all bbox") if bm_model else None

    # Biomarker zero-shot + trained — DOCTOR-ONLY
    print("\n  BIOMARKER EVAL — DOCTOR-ONLY")
    zs_doc = evaluate_biomarkers_zero_shot(model, loader_doc, ds_doc, gt_doc, processor, "ZS doctor-only")
    trained_doc = evaluate_biomarkers_trained(bm_model, loader_doc, ds_doc, gt_doc, bm_thrs,
                                              "v5 doctor-only") if bm_model else None

    # Stergem CSVurile temporare
    for tmp in [tmp_all, tmp_doc]:
        if os.path.exists(tmp):
            os.remove(tmp)

    # Salvam rezultatele
    results = {
        "model": "MedSigLIP_v13",
        "disease_zero_shot": disease_results,
        "biomarker_all_bbox": {"zero_shot": zs_all, "trained_v5": trained_all},
        "biomarker_doctor_only": {"zero_shot": zs_doc, "trained_v5": trained_doc},
    }
    os.makedirs(os.path.dirname(cfg.out_json), exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary final
    print("\n  SUMMARY FINAL")
    print("\n  Disease Zero-Shot:")
    for name, r in disease_results.items():
        print(f"    {name:<35} Acc={r['accuracy']}% | F1={r['f1_macro']}")

    print(f"\n  Biomarker — ALL bbox ({len(ds_all)} img):")
    if zs_all.get("macro_f1"):
        print(f"    Zero-shot:  Macro F1={zs_all['macro_f1']}")
    if trained_all and trained_all.get("macro_f1"):
        print(f"    Trained v5: Macro F1={trained_all['macro_f1']}")

    print(f"\n  Biomarker — DOCTOR-ONLY ({len(ds_doc)} img):")
    if zs_doc.get("macro_f1"):
        print(f"    Zero-shot:  Macro F1={zs_doc['macro_f1']}")
    if trained_doc and trained_doc.get("macro_f1"):
        print(f"    Trained v5: Macro F1={trained_doc['macro_f1']}")

    print(f"\n  Results: {cfg.out_json}")


if __name__ == "__main__":
    main()
