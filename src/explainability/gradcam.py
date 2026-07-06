"""
gradcam.py — GradCAM / EigenCAM / Rollout pt MedSigLIP v13

Vizualizare explainability cu heatmap-uri curate, aspect medical.

Detalii tehnice:
  - SigLIP NU are CLS token => nu facem slicing pe primul token
  - Target layer: layer_norm1 din layers[-2] (reduce artefactele tip grila)
  - FARA L2 norm inainte de logits (pastreaza magnitudinea gradientilor)
  - GaussianBlur + threshold pentru a elimina zgomotul din fundal
  - Suport adaugat pentru Attention Rollout (specific pentru ViT)

Output:
    experiments/figures/gradcam/
        eigencam_grid.png       <- grid cu toate imaginile (2-4 per clasa)
        eigencam_AMD_0.png      <- imagini individuale per clasa

Rulare:
    python src/explainability/gradcam.py --method eigencam
    python src/explainability/gradcam.py --method rollout
    python src/explainability/gradcam.py --image-path "poza.png" --method rollout
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.datasets.oct5k_medsiglip import OCT5kDataset
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


CHECKPOINT = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"
MODEL_PATH = "models/medsiglip-448"
OUTPUT_DIR = "experiments/figures/gradcam"
CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CAM_METHODS = {
    "gradcam":   GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam":  EigenCAM,
    "layercam":  LayerCAM,
    "rollout":   "rollout",
}


class GradCAMWrapper(torch.nn.Module):
    """Wrapper pt CAM — expune doar forward-ul vizual (image -> cls logits)."""
    def __init__(self, model: MedSigLIPMultiTask):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_pooled = self.model.encode_image(pixel_values)
        return self.model.classification_head(image_pooled)


# MODEL LOADING

def load_model(n_classes: int = 4) -> tuple:
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    remapped = {
        k.replace("sev_head.", "severity_head.")
         .replace("cls_head.", "classification_head.")
         .replace("fusion.attn_a2b.", "fusion.attn_a_to_b.")
         .replace("fusion.attn_b2a.", "fusion.attn_b_to_a."): v
        for k, v in state.items()
    }

    cls_hidden = 256
    for key in ["classification_head.1.weight", "cls_head.1.weight"]:
        if key in remapped:
            cls_hidden = remapped[key].shape[0]
            break

    model = MedSigLIPMultiTask(MODEL_PATH, n_classes=n_classes, cls_hidden=cls_hidden).to(DEVICE)
    model.load_state_dict(remapped, strict=False)
    model.eval()

    wrapper = GradCAMWrapper(model).to(DEVICE)
    return model, wrapper


# ATTENTION ROLLOUT (specific ViT)

def get_attention_rollout(model: MedSigLIPMultiTask, pixel_values: torch.Tensor) -> np.ndarray:
    """
    Calculeaza Attention Rollout extragand direct matricele de atentie din SigLIP.
    Ignora gradientii, folosind doar ponderile invatate de self-attention.
    """
    base_hf_model = model.backbone.base_model.model

    with torch.no_grad():
        outputs = base_hf_model.vision_model(pixel_values, output_attentions=True)
        attentions = outputs.attentions

    seq_len = attentions[0].shape[2]
    result = torch.eye(seq_len).to(pixel_values.device)

    for attention in attentions:
        attention_heads_fused = attention[0].mean(axis=0)
        attention_heads_fused += torch.eye(seq_len).to(pixel_values.device)
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(attention_heads_fused, result)

    mask = result.mean(dim=0)
    grid_size = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(grid_size, grid_size).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# RESHAPE TRANSFORM

def reshape_transform(tensor: torch.Tensor, height: int = 32, width: int = 32) -> torch.Tensor:
    n_patches = tensor.shape[1]
    h = w = int(n_patches ** 0.5)

    if n_patches == h * w + 1:
        tensor = tensor[:, 1:, :]
        n_patches = tensor.shape[1]
        h = w = int(n_patches ** 0.5)

    return tensor[:, :h * w, :].reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)


# PREPROCESSING

def auto_crop(img: Image.Image, threshold: int = 35) -> Image.Image:
    arr = np.array(img.convert("L"))
    mask = arr > threshold
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)

    if not (rows.any() and cols.any()):
        return img

    y1 = max(0, int(rows.argmax()) - 5)
    y2 = min(arr.shape[0], int(len(rows) - rows[::-1].argmax()) + 5)
    x1 = max(0, int(cols.argmax()) - 5)
    x2 = min(arr.shape[1], int(len(cols) - cols[::-1].argmax()) + 5)

    if (x2 - x1) > 50 and (y2 - y1) > 50:
        return img.crop((x1, y1, x2, y2))
    return img


def preprocess_image(path: str) -> Image.Image:
    pil = Image.open(path).convert("RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil = auto_crop(pil)
    pil = pil.resize((448, 448), Image.LANCZOS)
    return pil


# PROCESARE IMAGINE — forward + heatmap

def process_image(path, model, processor, cam_obj, method="eigencam"):
    pil = preprocess_image(path)
    rgb_resized = np.array(pil)
    rgb_float = np.float32(rgb_resized) / 255.0
    input_tensor = processor(images=pil, return_tensors="pt")["pixel_values"].to(DEVICE)

    if method == "rollout":
        grayscale_cam = get_attention_rollout(model, input_tensor)
        grayscale_cam = cv2.resize(grayscale_cam, (448, 448), interpolation=cv2.INTER_CUBIC)
    else:
        grayscale_cam = cam_obj(input_tensor=input_tensor, targets=None)[0]

    grayscale_cam = np.maximum(grayscale_cam, 0)

    if method == "rollout":
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (31, 31), 0)
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi
        grayscale_cam = grayscale_cam ** 2
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi
    else:
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (71, 71), 0)
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi
        grayscale_cam[grayscale_cam < 0.35] = 0
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi

    # Fara masca — doar threshold-ul de 0.35 curata fundalul
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        image_pooled = model.encode_image(input_tensor)
        cls_logits = model.classification_head(image_pooled)
        probs = torch.softmax(cls_logits, dim=1)[0]
        pred_class = CLASSES[probs.argmax().item()]
        confidence = probs.max().item() * 100
        severity_pct = model.severity_head(image_pooled).clamp(0, 1).item() * 100

    return rgb_resized, grayscale_cam, overlay, pred_class, confidence, severity_pct


# SELECTARE IMAGINI DE TEST

def get_test_images(processor: AutoProcessor, samples_per_class: int = 2) -> list[dict]:
    ds = OCT5kDataset(
        split_csv="data/oct5k/splits_v3/test.csv",
        split_json="data/OCT5k/medgemma_prompts_split_v2_27b.json",
        severity_json="data/oct5k/severity_scores_v2.json",
        processor=processor,
        mode="eval",
    )

    class_samples = {cls: [] for cls in ds.classes}
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
                "path": disk, "disease": disease, "label": ds.lbl_map[disease],
            })
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    return [sample for cls in CLASSES for sample in class_samples.get(cls, [])]


# PLOTARE

def plot_triplet(axes_row, rgb, heatmap, overlay, title_orig, title_cam, title_overlay):
    axes_row[0].imshow(rgb)
    axes_row[0].set_title(title_orig, fontsize=12)
    axes_row[0].axis("off")

    axes_row[1].imshow(heatmap, cmap="jet")
    axes_row[1].set_title(title_cam, fontsize=12)
    axes_row[1].axis("off")

    axes_row[2].imshow(overlay)
    axes_row[2].set_title(title_overlay, fontsize=12)
    axes_row[2].axis("off")


def save_individual(rgb, heatmap, overlay, disease, pred, conf, sev, method, idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_triplet(
        ax, rgb, heatmap, overlay,
        f"Original ({disease})",
        method.upper(),
        f"{pred} ({conf:.0f}%) Sev: {sev:.0f}%",
    )
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{method}_{disease}_{idx}.png", dpi=150)
    plt.close(fig)


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str,  default=None,       help="Cale imagine custom")
    parser.add_argument("--method",     type=str,  default="rollout",  choices=list(CAM_METHODS.keys()))
    parser.add_argument("--samples",    type=int,  default=2,          help="Imagini per clasa (2-4)")
    args = parser.parse_args()

    set_seed()

    print(f"  EXPLAINABILITY: {args.method.upper()}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model, cam_wrapper = load_model(n_classes=len(CLASSES))
    print(f"  Model incarcat pe {DEVICE}")

    # Construim cam_obj doar pt metodele non-rollout
    cam_obj = None
    if args.method != "rollout":
        base_hf_model = model.backbone.base_model.model
        target_layers = [base_hf_model.vision_model.encoder.layers[-2].layer_norm1]

        cam_obj = CAM_METHODS[args.method](
            model=cam_wrapper,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    if args.image_path:
        rgb, heatmap, overlay, pred, conf, sev = process_image(
            args.image_path, model, processor, cam_obj, args.method
        )
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        plot_triplet(ax, rgb, heatmap, overlay, "Original", args.method.upper(),
                     f"Pred: {pred} ({conf:.0f}%) | Sev: {sev:.0f}%")
        fig.tight_layout()
        out_path = f"{OUTPUT_DIR}/{args.method}_custom.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    else:
        images = get_test_images(processor, samples_per_class=args.samples)
        n_images = len(images)
        print(f"  Generam {n_images} imagini ({args.samples} per clasa)\n")

        fig, axes = plt.subplots(n_images, 3, figsize=(15, 4.5 * n_images))

        for i, info in enumerate(images):
            rgb, heatmap, overlay, pred, conf, sev = process_image(
                info["path"], model, processor, cam_obj, args.method
            )
            disease = info["disease"]
            correct = "✓" if pred == disease else "✗"

            plot_triplet(
                axes[i], rgb, heatmap, overlay,
                f"Original ({disease})",
                args.method.upper(),
                f"{correct} {pred} ({conf:.0f}%) | Sev: {sev:.0f}%",
            )

            save_individual(rgb, heatmap, overlay, disease, pred, conf, sev, args.method, i)
            print(f"  {disease}: pred={pred} ({conf:.0f}%) sev={sev:.0f}% {correct}")

        fig.suptitle(f"MedSigLIP v13 — {args.method.upper()} Explainability", fontsize=16, y=1.01)
        fig.tight_layout()
        grid_path = f"{OUTPUT_DIR}/{args.method}_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Grid salvat: {grid_path}")


if __name__ == "__main__":
    main()