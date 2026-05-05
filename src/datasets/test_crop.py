"""
Test vizual: auto-crop + Gaussian blur denoise
Arata 4 coloane: Original | Mask | Cropped | Cropped+Blur

Rulare:
    python src/datasets/test_crop.py
"""

import os
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import AutoProcessor
from src.datasets.oct5k_medsiglip import OCT5kDataset


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


def main():
    print("Loading dataset...")
    processor = AutoProcessor.from_pretrained("models/medsiglip-448")

    ds = OCT5kDataset(
        split_csv="data/oct5k/splits/test.csv",
        split_json="data/oct5k/medgemma_prompts_split.json",
        severity_json="data/oct5k/severity_scores.json",
        processor=processor,
        mode="eval",
    )

    # 2 imagini per clasa
    samples_per_class = 2
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
            class_samples[disease].append({"path": disk, "disease": disease})
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    all_samples = []
    for cls in ds.classes:
        all_samples.extend(class_samples[cls])

    n = len(all_samples)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))

    for i, sample in enumerate(all_samples):
        img = Image.open(sample["path"]).convert("RGB")
        cropped = auto_crop(img)
        blurred = cropped.filter(ImageFilter.GaussianBlur(radius=0.5))

        # mask
        arr = np.array(img.convert("L"))
        mask_vis = (arr > 35).astype(np.float32)

        # original
        axes[i, 0].imshow(np.array(img))
        axes[i, 0].set_title(f"Original ({sample['disease']})\n{img.size[0]}x{img.size[1]}")
        axes[i, 0].axis("off")

        # mask
        axes[i, 1].imshow(mask_vis, cmap="gray")
        axes[i, 1].set_title("Detection mask (th=35)")
        axes[i, 1].axis("off")

        # cropped
        axes[i, 2].imshow(np.array(cropped))
        axes[i, 2].set_title(f"Auto-cropped\n{cropped.size[0]}x{cropped.size[1]}")
        axes[i, 2].axis("off")

        # cropped + blur
        axes[i, 3].imshow(np.array(blurred))
        axes[i, 3].set_title(f"Cropped + Blur(0.5)\n{blurred.size[0]}x{blurred.size[1]}")
        axes[i, 3].axis("off")

    plt.suptitle("Pipeline: Original → Mask → Auto-Crop → Denoise", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = "experiments/figures/auto_crop_test.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # zoom comparison: cu blur vs fara blur (detaliu)
    sample = all_samples[0]
    img = Image.open(sample["path"]).convert("RGB")
    cropped = auto_crop(img)

    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))

    # crop o zona mica pt zoom
    w, h = cropped.size
    zoom = cropped.crop((w//4, h//4, 3*w//4, 3*h//4))

    axes2[0].imshow(np.array(zoom))
    axes2[0].set_title("Zoom: No blur")
    axes2[0].axis("off")

    for idx, radius in enumerate([0.3, 0.5, 1.0]):
        blurred_zoom = zoom.filter(ImageFilter.GaussianBlur(radius=radius))
        axes2[idx + 1].imshow(np.array(blurred_zoom))
        axes2[idx + 1].set_title(f"Zoom: Blur radius={radius}")
        axes2[idx + 1].axis("off")

    plt.suptitle(f"Blur Comparison (zoom on {sample['disease']})", fontsize=14)
    plt.tight_layout()
    plt.savefig("experiments/figures/blur_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: experiments/figures/blur_comparison.png")

    # stats
    print(f"\nCrop statistics:")
    for sample in all_samples:
        img = Image.open(sample["path"]).convert("RGB")
        cropped = auto_crop(img)
        pct = (cropped.size[0] * cropped.size[1]) / (img.size[0] * img.size[1]) * 100
        print(f"  {sample['disease']:8s}: {img.size} -> {cropped.size} ({pct:.0f}% kept)")


if __name__ == "__main__":
    main()