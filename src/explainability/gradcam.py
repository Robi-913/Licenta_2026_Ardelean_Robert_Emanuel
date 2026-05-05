"""
GradCAM / EigenCAM pt MedSigLIP v2

Vizualizare explainability cu heatmap-uri curate, aspect medical.

Fix-uri aplicate:
  - SigLIP NU are CLS token (fara slicing)
  - Target layer: layers[-2] (penultimul, mai multa info spatiala)
  - FARA L2 norm inainte de logits (pastreaza gradientii)
  - GaussianBlur(31,31) + threshold 25% (elimina zgomot fond)
  - EigenCAM default (mai robust pe ViT)

Output:
  experiments/figures/gradcam/
    eigencam_grid.png      <- grid 8-16 imagini (2-4 per clasa)
    + imagini individuale

Rulare:
    python src/explainability/gradcam.py                        (EigenCAM default)
    python src/explainability/gradcam.py --method gradcam       (GradCAM clasic)
    python src/explainability/gradcam.py --image-path "poza.png"
    python src/explainability/gradcam.py --samples 4            (4 per clasa)
"""

import argparse
import os
import sys
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.utils.seed import set_seed


# ---------- config ----------

MODEL_PATH = "models/medsiglip-448"
CHECKPOINT = "experiments/medsiglip_v3/ckpts/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]
OUTPUT_DIR = "experiments/figures/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- reshape transform ----------

def reshape_transform(tensor, height=32, width=32):
    """SigLIP NU are CLS token — detectam automat."""
    n_patches = tensor.shape[1]
    h = w = int(n_patches ** 0.5)

    if n_patches == h * w + 1:
        tensor = tensor[:, 1:, :]
        n_patches = tensor.shape[1]
        h = w = int(n_patches ** 0.5)

    tensor = tensor[:, :h * w, :]
    result = tensor.reshape(tensor.size(0), h, w, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


# ---------- model wrapper ----------

class MedSigLIPClassifier(nn.Module):

    def __init__(self, model_path, checkpoint_path, n_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        dim = self.backbone.config.vision_config.hidden_size

        self.cls_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )
        self.sev_head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.logit_scale = nn.Parameter(torch.ones([]))

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        own = self.state_dict()
        loaded = 0
        for name, param in state.items():
            if name in own and own[name].shape == param.shape:
                own[name].copy_(param)
                loaded += 1
        print(f"  Loaded {loaded}/{len(own)} params")

    def forward(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]
        else:
            emb = out
        return self.cls_head(emb)


# ---------- preprocessing ----------

def auto_crop(img, threshold=35):
    arr = np.array(img.convert("L"))
    mask = arr > threshold
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    if rows.any() and cols.any():
        y1 = int(rows.argmax())
        y2 = int(len(rows) - rows[::-1].argmax())
        x1 = int(cols.argmax())
        x2 = int(len(cols) - cols[::-1].argmax())
        pad = 5
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(arr.shape[0], y2 + pad)
        x2 = min(arr.shape[1], x2 + pad)
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            img = img.crop((x1, y1, x2, y2))
    return img


def preprocess_image(path):
    pil = Image.open(path).convert("RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil = auto_crop(pil)
    return pil


# ---------- smooth cam ----------

def smooth_cam(grayscale_cam, kernel_size=31, threshold_pct=0.35):
    """
    Publication-quality smoothing:
      1. GaussianBlur 31x31 — pete de caldura fluide
      2. Threshold 25% — elimina zgomot din fundal
      3. Re-normalizare 0-1
    """
    smoothed = cv2.GaussianBlur(grayscale_cam, (kernel_size, kernel_size), 0)

    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo > 1e-8:
        smoothed = (smoothed - lo) / (hi - lo)

    smoothed[smoothed < threshold_pct] = 0

    hi = smoothed.max()
    if hi > 1e-8:
        smoothed = smoothed / hi

    return smoothed


# ---------- get test images ----------

def get_test_images(samples_per_class=2):
    from src.datasets.oct5k_medsiglip import OCT5kDataset

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    ds = OCT5kDataset(
        split_csv="data/oct5k/splits/test.csv",
        split_json="data/oct5k/medgemma_prompts_split.json",
        severity_json="data/oct5k/severity_scores.json",
        processor=processor,
        mode="eval",
    )

    class_samples = {c: [] for c in ds.classes}
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)

    for idx in indices:
        row = ds.df.iloc[idx]
        disease = row["disease"]
        if len(class_samples[disease]) >= samples_per_class:
            continue
        disk = ds._locate(row["image_path"])
        if disk:
            class_samples[disease].append({
                "path": disk,
                "disease": disease,
                "label": ds.lbl_map[disease],
            })
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    images = []
    for cls in CLASSES:
        images.extend(class_samples.get(cls, []))
    return images


# ---------- process one image ----------

def process_image(path, model, processor, cam_obj):
    pil = preprocess_image(path)
    rgb_resized = cv2.resize(np.array(pil), (448, 448))
    rgb_float = np.float32(rgb_resized) / 255.0

    inputs = processor(images=pil, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(DEVICE)

    grayscale_cam = cam_obj(input_tensor=input_tensor, targets=None)[0, :]
    grayscale_cam = smooth_cam(grayscale_cam)

    # mascam fundalul negru — activam doar pe zona retiniana
    gray_img = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
    retina_mask = (gray_img > 35).astype(np.float32)
    retina_mask = cv2.GaussianBlur(retina_mask, (15, 15), 0)
    grayscale_cam = grayscale_cam * retina_mask

    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_cls = CLASSES[probs.argmax().item()]
        conf = probs.max().item() * 100

        emb = model.backbone.get_image_features(pixel_values=input_tensor)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif hasattr(emb, "last_hidden_state"):
            emb = emb.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=-1)
        sev = model.sev_head(emb).clamp(0, 1).item() * 100

    return rgb_resized, grayscale_cam, cam_image, pred_cls, conf, sev


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--method", type=str, default="eigencam",
                        choices=["gradcam", "gradcam++", "eigencam", "layercam"])
    parser.add_argument("--samples", type=int, default=2,
                        help="Samples per class (2-4)")
    args = parser.parse_args()

    set_seed()

    print(f"{'=' * 60}")
    print(f"  EXPLAINABILITY: {args.method.upper()}")
    print(f"  Target: layers[-2] | Blur: 31x31 | Threshold: 25%")
    print(f"  Samples per class: {args.samples}")
    print(f"{'=' * 60}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = MedSigLIPClassifier(MODEL_PATH, CHECKPOINT, n_classes=4)
    model = model.to(DEVICE)
    model.eval()
    print(f"  Model on {DEVICE}")

    target_layers = [model.backbone.vision_model.encoder.layers[-2]]

    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam": EigenCAM,
        "layercam": LayerCAM,
    }

    import matplotlib.pyplot as plt

    if args.image_path:
        cam_obj = methods[args.method](
            model=model, target_layers=target_layers,
            reshape_transform=reshape_transform,
        )
        rgb, hmap, overlay, pred, conf, sev = process_image(
            args.image_path, model, processor, cam_obj
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Original", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(hmap, cmap="jet")
        axes[1].set_title(f"{args.method.upper()} Heatmap", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {pred} ({conf:.0f}%) | Sev: {sev:.0f}%", fontsize=13)
        axes[2].axis("off")

        plt.tight_layout()
        out_path = f"{OUTPUT_DIR}/{args.method}_custom.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

    else:
        images = get_test_images(samples_per_class=args.samples)
        n_images = len(images)
        print(f"  {n_images} images ({args.samples} per class)\n")

        cam_obj = methods[args.method](
            model=model, target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

        fig, axes = plt.subplots(n_images, 3, figsize=(15, 4.5 * n_images))

        for i, img_info in enumerate(images):
            rgb, hmap, overlay, pred, conf, sev = process_image(
                img_info["path"], model, processor, cam_obj
            )
            disease = img_info["disease"]
            correct = "✓" if pred == disease else "✗"

            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f"Original ({disease})", fontsize=12)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(hmap, cmap="jet")
            axes[i, 1].set_title(f"{args.method.upper()}", fontsize=12)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(
                f"{correct} Pred: {pred} ({conf:.0f}%) | Sev: {sev:.0f}%",
                fontsize=12,
            )
            axes[i, 2].axis("off")

            # save individual
            fig_ind, ax_ind = plt.subplots(1, 3, figsize=(15, 5))
            ax_ind[0].imshow(rgb)
            ax_ind[0].set_title(f"Original ({disease})")
            ax_ind[0].axis("off")
            ax_ind[1].imshow(hmap, cmap="jet")
            ax_ind[1].set_title(args.method.upper())
            ax_ind[1].axis("off")
            ax_ind[2].imshow(overlay)
            ax_ind[2].set_title(f"{pred} ({conf:.0f}%) Sev:{sev:.0f}%")
            ax_ind[2].axis("off")
            fig_ind.tight_layout()
            fig_ind.savefig(f"{OUTPUT_DIR}/{args.method}_{disease}_{i}.png", dpi=150)
            plt.close(fig_ind)

            print(f"    {disease}: pred={pred} ({conf:.0f}%) sev={sev:.0f}% {correct}")

        plt.suptitle(
            f"MedSigLIP v2 — {args.method.upper()} Explainability",
            fontsize=16, y=1.01,
        )
        plt.tight_layout()
        grid_path = f"{OUTPUT_DIR}/{args.method}_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Grid: {grid_path}")
        print(f"  Individual: {OUTPUT_DIR}/")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()