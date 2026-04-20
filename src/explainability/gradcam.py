"""
GradCAM pentru MedSigLIP - vizualizare unde se uita modelul pe imaginile OCT

Genereaza heatmap-uri suprapuse pe imagini, cate una per clasa de boala.
Arata ce zone ale retinei activeaza cel mai mult modelul.

Output:
    experiments/figures/gradcam/          ← heatmaps individuale
    experiments/figures/gradcam_grid.png  ← grid comparativ per clasa

Rulare:
    python -m src.evaluation.gradcam
"""

import os
import sys
import gc
import json
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.seed import set_seed
from src.datasets.oct5k_medsiglip import OCT5kMedSigLIP


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    model_path = "models/medsiglip-448"
    checkpoint = "experiments/medsiglip_pipeline/ckpts/best.pth"

    splits_dir = "data/oct5k/splits"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    severity_json = "data/oct5k/severity_scores.json"
    test_csv = "data/oct5k/splits/test.csv"

    output_dir = "experiments/figures/gradcam"
    samples_per_class = 4  # cate imagini per clasa

    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL (copie simplificata pt GradCAM)
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


# ═══════════════════════════════════════════════════════════════════════
# GRADCAM
# ═══════════════════════════════════════════════════════════════════════

class GradCAM:
    """GradCAM pentru SigLIP Vision Transformer."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        # target layer = ultimul encoder layer din vision model
        if target_layer is None:
            vision_encoder = model.model.vision_model.encoder
            target_layer = vision_encoder.layers[-1]

        self.target_layer = target_layer

        # hook pt activari si gradienti
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output poate fi tuple sau tensor
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def generate(self, pixel_values, target_class=None):
        """Genereaza heatmap GradCAM."""
        self.model.zero_grad()

        # forward pass
        img_features = self.model.model.get_image_features(pixel_values=pixel_values)
        if hasattr(img_features, "pooler_output"):
            img_emb = img_features.pooler_output
        elif hasattr(img_features, "last_hidden_state"):
            img_emb = img_features.last_hidden_state[:, 0]
        else:
            img_emb = img_features

        # folosim cls_head pt a obtine logits
        cls_logits = self.model.cls_head(img_emb)

        if target_class is None:
            target_class = cls_logits.argmax(dim=1)

        # backward pe clasa target
        one_hot = torch.zeros_like(cls_logits)
        for i in range(len(target_class)):
            one_hot[i, target_class[i]] = 1

        cls_logits.backward(gradient=one_hot, retain_graph=True)

        # calcul GradCAM
        gradients = self.gradients  # [B, num_patches+1, dim]
        activations = self.activations  # [B, num_patches+1, dim]

        # skip CLS token (primul token)
        gradients = gradients[:, 1:, :]
        activations = activations[:, 1:, :]

        # weights = mean al gradientilor pe dimensiunea feature
        weights = gradients.mean(dim=-1, keepdim=True)  # [B, num_patches, 1]

        # weighted combination
        cam = (weights * activations).sum(dim=-1)  # [B, num_patches]
        cam = F.relu(cam)  # doar activari pozitive

        # reshape la grid spatial
        num_patches = cam.shape[1]
        h = w = int(num_patches ** 0.5)
        # taiem patch-urile extra daca nu e patrat perfect
        cam = cam[:, :h * w]
        cam = cam.view(-1, h, w)

        # normalize 0-1
        for i in range(cam.shape[0]):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max - cam_min > 0:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy(), cls_logits.detach().cpu()


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Suprapune heatmap pe imagine."""
    # resize heatmap la dimensiunea imaginii
    h, w = image.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # colormap
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB, fara alpha

    # overlay
    overlay = (1 - alpha) * image.astype(np.float32) / 255.0 + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay, heatmap_resized


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    print(f"{'=' * 70}")
    print("  GRADCAM: MedSigLIP Attention Visualization")
    print(f"{'=' * 70}")

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # incarcam modelul
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    nc = ckpt.get("num_classes", 4)
    classes = ckpt.get("classes", ["AMD", "DME", "DRUSEN", "NORMAL"])

    model = MedSigLIPMultiTask(cfg.model_path, num_classes=nc)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device)
    model.eval()

    # GradCAM
    gradcam = GradCAM(model)

    # dataset
    test_ds = OCT5kMedSigLIP(
        split_csv=cfg.test_csv,
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=processor,
        mode="eval",
    )

    # selectam samples per clasa
    class_indices = {c: [] for c in classes}
    for idx in range(len(test_ds)):
        row = test_ds.df.iloc[idx]
        disease = row["disease"]
        if len(class_indices[disease]) < cfg.samples_per_class:
            class_indices[disease].append(idx)

    # generam GradCAM per sample
    all_results = {c: [] for c in classes}

    for cls_name, indices in class_indices.items():
        print(f"\n  Procesare {cls_name}: {len(indices)} samples")
        for idx in indices:
            sample = test_ds[idx]
            pv = sample["pixel_values"].unsqueeze(0).to(cfg.device)

            # generam heatmap
            heatmap, logits = gradcam.generate(pv)
            pred_class = logits.argmax(1).item()
            pred_name = classes[pred_class]
            confidence = torch.softmax(logits, dim=1).max().item()

            # imaginea originala
            row = test_ds.df.iloc[idx]
            disk_path = test_ds._find_image(row["image_path"])
            orig_img = np.array(Image.open(disk_path).convert("RGB"))

            # overlay
            overlay, hmap = overlay_heatmap(orig_img, heatmap[0])

            # severity
            with torch.no_grad():
                img_emb = model.model.get_image_features(pixel_values=pv)
                if hasattr(img_emb, "pooler_output"):
                    img_emb = img_emb.pooler_output
                elif hasattr(img_emb, "last_hidden_state"):
                    img_emb = img_emb.last_hidden_state[:, 0]
                sev = model.severity_head(img_emb).item() * 100

            all_results[cls_name].append({
                "original": orig_img,
                "heatmap": hmap,
                "overlay": overlay,
                "pred": pred_name,
                "confidence": confidence,
                "severity": sev,
                "true_class": cls_name,
            })

            # salveaza individual
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original ({cls_name})")
            axes[0].axis("off")

            axes[1].imshow(hmap, cmap="jet")
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Pred: {pred_name} ({confidence*100:.0f}%) | Sev: {sev:.0f}%")
            axes[2].axis("off")

            plt.tight_layout()
            fname = f"{cfg.output_dir}/{cls_name}_{idx}.png"
            plt.savefig(fname, dpi=150)
            plt.close()

    # grid comparativ: 4 clase x samples_per_class
    print(f"\n  Generare grid comparativ...")
    n_cols = cfg.samples_per_class
    n_rows = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for row_idx, cls_name in enumerate(classes):
        results = all_results[cls_name]
        for col_idx in range(min(n_cols, len(results))):
            r = results[col_idx]
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            ax.imshow(r["overlay"])
            ax.set_title(
                f"True: {r['true_class']}\n"
                f"Pred: {r['pred']} ({r['confidence']*100:.0f}%)\n"
                f"Severity: {r['severity']:.0f}%",
                fontsize=10,
            )
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(cls_name, fontsize=14, fontweight="bold", rotation=0,
                             labelpad=60, va="center")

    plt.suptitle("MedSigLIP GradCAM — Attention per Disease Class", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"experiments/figures/gradcam_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  GradCAM salvat:")
    print(f"    Individual: {cfg.output_dir}/")
    print(f"    Grid: experiments/figures/gradcam_grid.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()