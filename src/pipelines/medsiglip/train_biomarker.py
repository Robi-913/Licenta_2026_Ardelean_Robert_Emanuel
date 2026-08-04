import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.model.medsiglip import MedSigLIPMultiTask, BiomarkerHeadsV5
from src.utils.seed import set_seed


class Config:
    ms_ckpt = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"
    model_path = "models/medsiglip-448"
    splits_dir = "data/oct5k/splits_biomk"
    master_json = "data/oct5k/metadata_v2/_master.json"
    save_dir = "experiments/biomarker_heads_v5"

    # LoRA config — IDENTIC cu v15, altfel load_state_dict da eroare
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05

    batch_size = 16
    epochs = 100
    lr_heads = 5e-4  # doar head-urile se antreneaza, backbone frozen
    wd = 0.01
    grad_clip = 1.0
    patience = 20

    # Focal loss — alpha mic pt a penaliza mai mult false positives pe clase rare
    focal_alpha = 0.25
    focal_gamma = 2.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    workers = 0

    @property
    def ckpt_dir(self):
        return f"{self.save_dir}/ckpts"


cfg = Config()
os.makedirs(cfg.ckpt_dir, exist_ok=True)

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]
N_BM = len(BIOMARKERS)
BM2IDX = {bm: i for i, bm in enumerate(BIOMARKERS)}

# Normalizam numele biomarkerilor pt matching robust (spatii, underscore, case)
_BM_NORMALIZED = {bm.lower().replace(" ", "").replace("_", ""): bm for bm in BIOMARKERS}


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


_IMG_SEARCH_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


def locate_image(meta: dict) -> str | None:
    """
    Incearca sa gaseasca fisierul de imagine pe disk.
    Ordinea: calea absoluta din meta -> cai relative in directoarele cunoscute.
    Returneaza None daca nu gaseste nimic.
    """
    # Prima optiune: calea absoluta salvata in metadata
    disk_path = meta.get("image_disk_path", "")
    if disk_path and Path(disk_path).exists():
        return str(disk_path)

    # A doua optiune: cautam relativ in directoarele IMG_DIRS cu mai multe extensii
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in _IMG_SEARCH_DIRS:
        candidate = Path(base) / rel
        if candidate.exists():
            return str(candidate)
        for ext in [".png", ".jpeg", ".jpg"]:
            with_ext = candidate.with_suffix(ext)
            if with_ext.exists():
                return str(with_ext)

    return None


class BiomarkerDataset(Dataset):
    """
    Dataset cu imagini adnotate doar de doctor (bbox_source == 'doctor').
    YOLO silver labels excluse — ground truth curat.

    Fiecare sample are:
      - pixel_values: imaginea procesata pt backbone MedSigLIP
      - labels:       tensor binar [N_BM] — 1 daca biomarkerul e prezent
    """

    def __init__(self, image_paths: list, metadata_dict: dict, processor: AutoProcessor, mode: str = "train"):
        self.processor = processor
        self.mode = mode
        self.samples = self._build_samples(image_paths, metadata_dict)

        print(f"  BiomarkerDataset [{mode}]: {len(self.samples)} imagini (doctor-only)")
        self._print_label_distribution()

    def _build_samples(self, image_paths: list, metadata_dict: dict) -> list:
        """Filtreaza imaginile valide si construieste labelele binare per biomarker."""
        samples = []
        for path in image_paths:
            meta = metadata_dict.get(path)

            # Filtram: trebuie sa aiba bbox si sa fie doctor-only
            if meta is None or not meta.get("has_bounding_boxes"):
                continue
            if meta.get("bbox_source", "doctor") != "doctor":
                continue

            oct_path = locate_image(meta)
            if oct_path is None:
                continue

            # Construim vectorul binar de label-uri din lista de leziuni
            labels = torch.zeros(N_BM)
            seen = set()
            for lesion in meta.get("lesions", []):
                bm_key = _BM_NORMALIZED.get(_normalize(lesion.get("class", "")))
                if bm_key and bm_key not in seen:
                    labels[BM2IDX[bm_key]] = 1.0
                    seen.add(bm_key)

            samples.append({"oct_path": oct_path, "labels": labels, "image_path": path})

        return samples

    def _print_label_distribution(self) -> None:
        if not self.samples:
            return
        all_labels = torch.stack([s["labels"] for s in self.samples])
        print(f"  Pozitive per biomarker:")
        for i, bm in enumerate(BIOMARKERS):
            n_pos = int(all_labels[:, i].sum())
            print(f"    {bm:<25}: {n_pos}/{len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s["oct_path"]).convert("RGB")

        # Blur usor pt a reduce zgomotul de speckle specific OCT
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        if self.mode == "train":
            img = self._augment(img)

        px = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": px["pixel_values"].squeeze(0),
            "labels": s["labels"],
        }

    @staticmethod
    def _augment(img: Image.Image) -> Image.Image:
        """Augmentari usoare — imaginile OCT sunt sensibile, nu exageram."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(5),
        ])(img)


def collate_bm(batch: list) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class FocalLoss(nn.Module):
    """
    Focal Loss pentru clasificare binara multi-label.
    Penalizeaza mai mult exemplele usor de clasificat gresit (hard negatives)
    si mai putin pe cele usor de clasificat corect.

    FL = alpha_t * (1 - p_t)^gamma * BCE

    Util pt clasele rare din biomarkeri — dataset-ul e dezechilibrat.

    :param alpha: pondereaza pozitivele vs negativele (0.25 = penalizeaza mai mult FP)
    :param gamma: focalizare pe exemple grele (0 = BCE standard, 2 = standard focal)
    :param pos_weight: weight per clasa pt a compensa dezechilibrul (n_neg/n_pos per biomarker)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE de baza — cu pos_weight daca avem
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        # p_t = probabilitatea clasei corecte (pt pozitive: p, pt negative: 1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # (1 - p_t)^gamma => exemple bine clasificate (p_t aproape 1) primesc weight mic
        focal_weight = (1 - p_t) ** self.gamma

        # alpha_t: weight diferit pt pozitive vs negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return (alpha_t * focal_weight * bce).mean()


def compute_pos_weights(samples: list) -> torch.Tensor:
    """
    Calculeaza n_neg/n_pos per biomarker.
    Dat ca pos_weight la FocalLoss => penalizeaza mai mult biomarkerii rari.
    """
    all_labels = torch.stack([s["labels"] for s in samples])
    n = len(samples)
    weights = []
    print("\n  Class weights (n_neg / n_pos):")
    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(all_labels[:, i].sum())
        n_neg = n - n_pos
        w = n_neg / n_pos if n_pos > 0 else 1.0
        weights.append(w)
        print(f"    {bm:<25}: pos={n_pos}, neg={n_neg}, w={w:.1f}")
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def optimize_thresholds(model: BiomarkerHeadsV5, loader: DataLoader) -> list[float]:
    """
    Cauta threshold-ul optim per biomarker pe setul de validare.
    Default 0.5 poate sa nu fie optim cand clasele sunt dezechilibrate.
    Parcurgem [0.1, 0.9] cu pas 0.05 si alegem threshold-ul cu cel mai mare F1.
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        pv = batch["pixel_values"].to(cfg.device)
        with autocast(cfg.device, enabled=cfg.use_amp):
            logits = model(pv)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    thresholds = []
    print("\n  Threshold optimization per biomarker:")
    for i, bm in enumerate(BIOMARKERS):
        if int(labels[:, i].sum()) == 0:
            thresholds.append(0.5)  # niciun pozitiv => threshold default
            continue

        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] > thr).astype(float)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        thresholds.append(round(best_thr, 2))
        print(f"    {bm:<25}: thr={best_thr:.2f} -> F1={best_f1:.3f}")

    return thresholds


@torch.no_grad()
def evaluate(
        model: BiomarkerHeadsV5,
        loader: DataLoader,
        thresholds: list[float] | None = None,
) -> dict:
    """
    Evalueaza modelul si returneaza F1, Precision, Recall per biomarker + Macro F1.
    Daca thresholds e None, foloseste 0.5 pentru toti.
    """
    model.eval()
    if thresholds is None:
        thresholds = [0.5] * N_BM

    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        with autocast(cfg.device, enabled=cfg.use_amp):
            logits = model(pv)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["labels"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    results = {}
    f1_scores = []
    for i, bm in enumerate(BIOMARKERS):
        n_pos = int(labels[:, i].sum())
        if n_pos == 0:
            results[bm] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_pos": 0}
            continue

        preds = (probs[:, i] > thresholds[i]).astype(float)
        results[bm] = {
            "f1": round(float(f1_score(labels[:, i], preds, zero_division=0)), 4),
            "precision": round(float(precision_score(labels[:, i], preds, zero_division=0)), 4),
            "recall": round(float(recall_score(labels[:, i], preds, zero_division=0)), 4),
            "n_pos": n_pos,
        }
        f1_scores.append(results[bm]["f1"])

    results["macro_f1"] = round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4)
    return results


def _free_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_backbone_from_v15(ckpt_path: str) -> MedSigLIPMultiTask:
    """
    Incarca backbone-ul MedSigLIPMultiTask din checkpointul v15.
    Filtram doar cheile care incep cu 'backbone.' si le incarcam strict.
    Restul (heads, fusion, logit_scale) nu ne intereseaza — vor fi ignorati.
    """
    print(f"\n  Loading backbone + LoRA from {ckpt_path}...")

    # Cream modelul cu aceiasi parametri LoRA ca v15 — altfel cheile nu se potrivesc
    model = MedSigLIPMultiTask(
        cfg.model_path,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # final_with_probe.pth e salvat direct ca state_dict, nu ca {"model": ...}
    state = ckpt.get("model", ckpt)

    # Filtram starea: vrem doar parametrii backbone-ului, nu head-urile sau fusion

    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state, strict=True)

    print("  Backbone + LoRA incarcate cu succes!")

    return model


def main():
    print("  BIOMARKER HEADS v5 — Doctor-only + LoRA backbone v13")
    print(f"  Epochs: {cfg.epochs} | LR heads: {cfg.lr_heads}")

    set_seed()

    # Incarcam metadata din master JSON
    with open(cfg.master_json, "r", encoding="utf-8") as f:
        master = json.load(f)
    # Convertim lista in dict keyed pe image_path pt lookup O(1)
    metadata_dict = {m["image_path"]: m for m in master}

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # Filtram doar imaginile cu bbox din split-urile corespunzatoare
    train_csv = pd.read_csv(f"{cfg.splits_dir}/train.csv")
    val_csv = pd.read_csv(f"{cfg.splits_dir}/val.csv")

    train_paths = train_csv[train_csv["has_bbox"] == True]["image_path"].tolist()
    val_paths = val_csv[val_csv["has_bbox"] == True]["image_path"].tolist()

    train_ds = BiomarkerDataset(train_paths, metadata_dict, processor, mode="train")
    val_ds = BiomarkerDataset(val_paths, metadata_dict, processor, mode="eval")

    if not train_ds.samples:
        raise RuntimeError("Nu s-au gasit imagini doctor cu bbox in train split!")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_bm,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_bm,
    )

    # Pos weights pentru focal loss — compenseaza dezechilibrul de clase
    pos_weights = compute_pos_weights(train_ds.samples)
    loss_fn = FocalLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
        pos_weight=pos_weights.to(cfg.device),
    )

    # Incarcam backbone-ul din v15 si construim modelul de biomarkeri
    base_model = load_backbone_from_v15(cfg.backbone_ckpt)
    model = BiomarkerHeadsV5(backbone=base_model, n_biomarkers=N_BM).to(cfg.device)

    # Optimizam DOAR head-urile — backbone frozen din BiomarkerHeadsV5.__init__
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=cfg.lr_heads, weight_decay=cfg.wd)
    scaler = GradScaler(cfg.device, enabled=cfg.use_amp)

    best_f1 = 0.0
    wait = 0

    print(f"\n{'=' * 60}")
    for epoch in range(cfg.epochs):
        # Backbone in eval mode chiar daca e frozen — dezactiveaza BatchNorm si Dropout din el
        model.train()
        model.backbone.eval()

        tot_loss, steps = 0.0, 0

        for batch in tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{cfg.epochs}", leave=False):
            pv = batch["pixel_values"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            with autocast(cfg.device, enabled=cfg.use_amp):
                logits = model(pv)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            tot_loss += loss.item()
            steps += 1

        metrics = evaluate(model, val_loader)
        macro_f1 = metrics["macro_f1"]

        improved = macro_f1 > best_f1
        if improved:
            best_f1 = macro_f1
            wait = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "best_f1": best_f1, "metrics": metrics,
                "biomarkers": BIOMARKERS, "version": "v5",
            }, f"{cfg.ckpt_dir}/best.pth")

        else:
            wait += 1

        marker = f"  Best: {best_f1:.4f}" if improved else f"  ({wait}/{cfg.patience})"
        print(f"  Epoch {epoch + 1}: Loss={tot_loss / steps:.4f} | Macro F1={macro_f1:.4f}{marker}")

        if wait >= cfg.patience:
            print(f"  Early stopping la epoch {epoch + 1}")
            break

        _free_mem()

    print("  THRESHOLD OPTIMIZATION pe best checkpoint...")

    best_ckpt = torch.load(f"{cfg.ckpt_dir}/best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    thresholds = optimize_thresholds(model, val_loader)
    metrics_opt = evaluate(model, val_loader, thresholds)

    print(f"\n  Macro F1 (thr=0.5):       {best_f1:.4f}")
    print(f"  Macro F1 (thr optimizat):  {metrics_opt['macro_f1']:.4f}")
    print()
    for bm in BIOMARKERS:
        m = metrics_opt[bm]
        if m["n_pos"] > 0:
            print(f"  {bm:<25}: F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    # Salvam checkpoint final cu threshold-urile optime incluse
    torch.save({
        "epoch": best_ckpt["epoch"],
        "model": model.state_dict(),
        "best_f1": best_f1,
        "best_f1_opt": metrics_opt["macro_f1"],
        "thresholds": thresholds,
        "metrics": metrics_opt,
        "biomarkers": BIOMARKERS,
        "version": "v5",
    }, f"{cfg.ckpt_dir}/best.pth")

    print(f"  DONE! Macro F1 = {metrics_opt['macro_f1']:.4f}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Saved: {cfg.ckpt_dir}/best.pth")


if __name__ == "__main__":
    main()
