"""
gradio_app.py — MedSigLIP v13 OCT Analyzer

Rulare:
    python src/demo/gradio_app.py
    -> http://localhost:7860

Pipeline:
  - EigenCAM pe layers[-2] pt explainability
  - CrossAttentionFusion pt retrieval contrastiv
  - Clasificare boala + estimare severitate
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.medsiglip import MedSigLIPMultiTask


MDL_PATH   = "model/medsiglip-448"
CKPT_PATH  = "experiments/medsiglip_v13/ckpts/final_with_probe.pth"
SPLIT_JSON = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
SEV_JSON   = "data/OCT5k/severity_scores_v2.json"

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CLS_NAMES = ["AMD", "DME", "DRUSEN", "NORMAL"]

# Cuvinte cheie pt detectia bolii din promptul structural
_DISEASE_KEYWORDS = {
    "age-related":        "AMD",
    "macular degeneration": "AMD",
    "amd":                "AMD",
    "diabetic":           "DME",
    "dme":                "DME",
    "drusen":             "DRUSEN",
    "normal":             "NORMAL",
}


# MODEL + CAM LOADING

def _load_model() -> tuple[MedSigLIPMultiTask, list[str]]:
    ckpt    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state   = ckpt.get("model", ckpt)
    nc      = ckpt.get("num_classes", 4) if isinstance(ckpt, dict) else 4
    classes = ckpt.get("classes", CLS_NAMES) if isinstance(ckpt, dict) else CLS_NAMES

    model = MedSigLIPMultiTask(MDL_PATH, n_classes=nc)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    return model, classes


def _build_cam(model: MedSigLIPMultiTask) -> EigenCAM:
    target_layers = [model.backbone.base_model.model.vision_model.encoder.layers[-2]]
    return EigenCAM(model=model, target_layers=target_layers, reshape_transform=_reshape_transform)


# CAM HELPERS

def _reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    pytorch-grad-cam asteapta [B, C, H, W] dar ViT produce [B, seq_len, dim].
    SigLIP NU are CLS token — convertim direct la grila 2D.
    """
    n_patches = tensor.shape[1]
    h = w = int(n_patches ** 0.5)
    if n_patches == h * w + 1:  # detectie automata CLS token daca exista
        tensor    = tensor[:, 1:, :]
        n_patches = tensor.shape[1]
        h = w     = int(n_patches ** 0.5)
    return tensor[:, :h * w, :].reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)


def _smooth_cam(grayscale_cam: np.ndarray, kernel_size: int = 31, threshold: float = 0.35) -> np.ndarray:
    """Blur + threshold + renormalizare pt heatmap curat fara pixeli izolati."""
    smoothed = cv2.GaussianBlur(grayscale_cam, (kernel_size, kernel_size), 0)
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo > 1e-8:
        smoothed = (smoothed - lo) / (hi - lo)
    smoothed[smoothed < threshold] = 0.0
    hi = smoothed.max()
    if hi > 1e-8:
        smoothed /= hi
    return smoothed


# PREPROCESSING

def _auto_crop(img: Image.Image, threshold: int = 35) -> Image.Image:
    """Taie marginile negre din imaginile OCT prin detectia zonei retiniene."""
    arr  = np.array(img.convert("L"))
    mask = arr > threshold
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    if rows.any() and cols.any():
        y1 = max(0, int(rows.argmax()) - 5)
        y2 = min(arr.shape[0], int(len(rows) - rows[::-1].argmax()) + 5)
        x1 = max(0, int(cols.argmax()) - 5)
        x2 = min(arr.shape[1], int(len(cols) - cols[::-1].argmax()) + 5)
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            return img.crop((x1, y1, x2, y2))
    return img


def _preprocess(image_array: np.ndarray) -> Image.Image:
    pil = Image.fromarray(image_array).convert("RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil = _auto_crop(pil)
    return pil


# RETRIEVAL DATABASE

def _detect_disease(prompt_a: str) -> str:
    """Detecteaza clasa de boala din primul prompt (structural) prin keyword matching."""
    p = prompt_a.lower()
    for kw, cls in _DISEASE_KEYWORDS.items():
        if kw in p:
            return cls
    return "UNKNOWN"


def _build_retrieval_db(
    model: MedSigLIPMultiTask,
    processor: AutoProcessor,
    split_json: str,
    sev_json: str,
) -> tuple[list[dict], torch.Tensor]:
    """
    Pre-computa embedding-urile fuzionate pt toate cazurile din baza de date.
    Returneaza (lista de metadate, matrice de embeddings [N, dim]).
    """
    with open(split_json, "r", encoding="utf-8") as f:
        split_raw = json.load(f)
    with open(sev_json, "r", encoding="utf-8") as f:
        sev_lookup = {
            x["image_path"].replace("/", "\\"): x
            for x in json.load(f)
            if x.get("severity_valid")
        }

    db = []
    for img_path, item in split_raw.items():
        pa, pb = item.get("a", ""), item.get("b", "")
        if not pa or not pb:
            continue

        def _tok(text):
            enc  = processor.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc.get("attention_mask", torch.ones_like(ids)).to(DEVICE)
            return ids, mask

        with torch.no_grad():
            ea     = model.encode_text(*_tok(pa))
            eb     = model.encode_text(*_tok(pb))
            fused  = model.fusion(ea, eb)

        path_norm = img_path.replace("/", "\\")
        sev_info  = sev_lookup.get(path_norm, {})

        db.append({
            "emb":      fused.cpu(),
            "prompt_a": pa,
            "prompt_b": pb,
            "disease":  _detect_disease(pa),
            "path":     img_path,
            "sev":      sev_info.get("severity_percent"),
        })

    all_embs = torch.cat([r["emb"] for r in db])
    return db, all_embs


# INITIALIZARE GLOBALA (o singura data la pornire)

print("  Loading model...")
_processor      = AutoProcessor.from_pretrained(MDL_PATH)
_model, _classes = _load_model()
_cam             = _build_cam(_model)

print("  Building retrieval database...")
_ret_db, _all_txt_embs = _build_retrieval_db(_model, _processor, SPLIT_JSON, SEV_JSON)
print(f"  Retrieval DB: {len(_ret_db)} entries | Ready!")


# SEVERITY LABEL

def _severity_label(pct: float) -> str:
    if pct < 15: return "Minimal"
    if pct < 30: return "Mild"
    if pct < 50: return "Moderate"
    if pct < 70: return "Significant"
    if pct < 85: return "Severe"
    return "Critical"


# ANALIZA IMAGINE

def analyze(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "Upload an OCT image to analyze."

    pil = _preprocess(image)
    pv  = _processor(images=pil, return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.no_grad():
        # features brute pt head-uri (ne-normalizate)
        image_pooled = _model.encode_image(pv)
        cls_logits   = _model.classification_head(image_pooled)
        sev_pct      = _model.severity_head(image_pooled).clamp(0, 1).item() * 100

        # embedding normalizat pt retrieval
        img_emb_norm = F.normalize(image_pooled, p=2, dim=-1)

    probs    = torch.softmax(cls_logits, dim=1)[0]
    pred_cls = _classes[probs.argmax().item()]
    conf     = probs.max().item() * 100
    per_cls  = {_classes[i]: float(probs[i]) for i in range(len(_classes))}

    # EigenCAM — heatmap pe zona retiniana
    rgb_resized   = cv2.resize(np.array(pil), (448, 448))
    rgb_float     = np.float32(rgb_resized) / 255.0
    grayscale_cam = _cam(input_tensor=pv, targets=None)[0]
    grayscale_cam = _smooth_cam(grayscale_cam)
    retina_mask   = cv2.GaussianBlur((cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY) > 35).astype(np.float32), (15, 15), 0)
    grayscale_cam *= retina_mask
    overlay       = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Retrieval top 3 cazuri similare
    with torch.no_grad():
        sim = (img_emb_norm.cpu() @ _all_txt_embs.T).squeeze(0)
    top = sim.topk(3)

    matches = ""
    for rank, (score, idx) in enumerate(zip(top.values, top.indices)):
        r = _ret_db[idx.item()]
        matches += f"\n{'─' * 50}\n"
        matches += f"Match #{rank + 1} (similarity: {score.item():.3f})\n"
        matches += f"Disease: {r['disease']}"
        if r["sev"] is not None:
            matches += f" | Severity: {r['sev']:.0f}%"
        matches += f"\n\nStructure:\n{r['prompt_a']}\n"
        matches += f"\nLesions:\n{r['prompt_b']}\n"

    bar_sev = int(sev_pct / 2)
    report  = f"""
DIAGNOSIS: {pred_cls}
Confidence: {conf:.1f}%

SEVERITY: {sev_pct:.1f}% ({_severity_label(sev_pct)})
{'█' * bar_sev}{'░' * (50 - bar_sev)}

CONFIDENCE PER CLASS:
  AMD:    {'█' * int(per_cls.get('AMD',    0) * 50)} {per_cls.get('AMD',    0) * 100:.1f}%
  DME:    {'█' * int(per_cls.get('DME',    0) * 50)} {per_cls.get('DME',    0) * 100:.1f}%
  DRUSEN: {'█' * int(per_cls.get('DRUSEN', 0) * 50)} {per_cls.get('DRUSEN', 0) * 100:.1f}%
  NORMAL: {'█' * int(per_cls.get('NORMAL', 0) * 50)} {per_cls.get('NORMAL', 0) * 100:.1f}%

SIMILAR CASES FROM DATABASE:
{matches}
"""
    return overlay, report


# GRADIO UI

def main():
    import gradio as gr

    with gr.Blocks(title="MedSigLIP v13 OCT Analyzer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # MedSigLIP v13 — Retinal OCT Analyzer
        ### MedGemma 27B Prompts + Cross-Attention Fusion + Multi-Task Learning

        Upload a retinal OCT scan to get:
        - **Disease Classification** (AMD / DME / DRUSEN / NORMAL)
        - **Severity Estimation** (0-100%)
        - **EigenCAM Heatmap** (where the model looks)
        - **Similar Cases** from the database with clinical descriptions
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="Upload OCT Scan", type="numpy")
                btn     = gr.Button("Analyze", variant="primary", size="lg")
                gr.Markdown("Upload any retinal OCT B-scan image (grayscale or RGB).")

            with gr.Column(scale=1):
                out_img = gr.Image(label="EigenCAM Attention Map", type="numpy")

        with gr.Row():
            out_report = gr.Textbox(label="Analysis Report", lines=30, max_lines=50)

        btn.click(fn=analyze, inputs=[inp_img], outputs=[out_img, out_report])

        gr.Markdown("""
        ---
        **Thesis Project** — Retinal OCT Disease Classification using MedSigLIP Multi-Task Learning

        *Pipeline: MedGemma 27B IT → Gemini Flash-Lite (split) → MedSigLIP v13 (cross-attention + dual contrastive)*

        Explainability: EigenCAM (layers[-2], blur=31, threshold=35%)

        This is a research tool, not a medical diagnostic device.
        """)

    app.launch(share=False, server_name="localhost", server_port=7860)


if __name__ == "__main__":
    main()