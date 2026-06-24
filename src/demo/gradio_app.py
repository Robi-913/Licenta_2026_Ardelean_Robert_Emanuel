"""
Gradio Demo: MedSigLIP v7 OCT Analyzer
EigenCAM + Cross-Attention Fusion + Dual Contrastive

Rulare:
    python src/demo/gradio_app.py
    -> http://localhost:7860
"""

import os
import sys
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoModel, AutoProcessor

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------- config ----------

MDL_PATH   = "models/medsiglip-448"
CKPT_PATH  = "experiments/medsiglip_v7/ckpts/best.pth"
SPLIT_JSON = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
SEV_JSON   = "data/OCT5k/severity_scores_v2.json"

DEV       = "cuda" if torch.cuda.is_available() else "cpu"
CLS_NAMES = ["AMD", "DME", "DRUSEN", "NORMAL"]


# ---------- cross-attention fusion ----------

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        # v7 checkpoint are attn_a2b/attn_b2a — rename la load
        self.attn_oct2mask = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_mask2oct = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )

    def forward(self, emb_a, emb_b):
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)
        attn_a, _ = self.attn_oct2mask(query=a, key=b, value=b)
        attn_b, _ = self.attn_mask2oct(query=b, key=a, value=a)
        attn_a = attn_a.squeeze(1)
        attn_b = attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


# ---------- model ----------

class MedSigLIPMultiTask(nn.Module):

    def __init__(self, model_path, n_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        dim = self.backbone.config.vision_config.hidden_size

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

    def forward(self, pixel_values):
        """Forward pt EigenCAM — raw logits."""
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]
        else:
            emb = out
        return self.cls_head(emb)


# ---------- eigencam helpers ----------

def reshape_transform(tensor, height=32, width=32):
    """SigLIP NU are CLS token."""
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


def smooth_cam(grayscale_cam, kernel_size=31, threshold_pct=0.35):
    smoothed = cv2.GaussianBlur(grayscale_cam, (kernel_size, kernel_size), 0)
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo > 1e-8:
        smoothed = (smoothed - lo) / (hi - lo)
    smoothed[smoothed < threshold_pct] = 0
    hi = smoothed.max()
    if hi > 1e-8:
        smoothed = smoothed / hi
    return smoothed


# ---------- auto crop ----------

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


# ---------- normalize path pt lookup ----------

def norm_path(p):
    """Normalizeaza path la backslash consistent pt lookup."""
    return p.replace("/", "\\")


# ---------- load model ----------

print("Loading model...")
proc = AutoProcessor.from_pretrained(MDL_PATH)

ckpt    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
nc      = ckpt.get("num_classes", 4)
classes = ckpt.get("classes", CLS_NAMES)

model = MedSigLIPMultiTask(MDL_PATH, n_classes=nc)

# rename keys: attn_a2b -> attn_oct2mask (v7 checkpoint compat)
state_dict = {}
for k, v in ckpt["model"].items():
    if "fusion.attn_a2b" in k:
        k = k.replace("fusion.attn_a2b", "fusion.attn_oct2mask")
    elif "fusion.attn_b2a" in k:
        k = k.replace("fusion.attn_b2a", "fusion.attn_mask2oct")
    state_dict[k] = v
model.load_state_dict(state_dict)
model = model.to(DEV)
model.eval()

# EigenCAM
target_layers = [model.backbone.vision_model.encoder.layers[-2]]
cam_gen = EigenCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)

# ---------- load retrieval DB ----------

print("Loading retrieval database...")

# v2 format: dict cu image_path ca key, campuri "a" si "b"
with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split_raw = json.load(f)

# severity — keyed by image_path (normalizeaza paths)
with open(SEV_JSON, "r", encoding="utf-8") as f:
    sev_list = json.load(f)

sev_lookup = {}
for x in sev_list:
    if x.get("severity_valid"):
        sev_lookup[norm_path(x["image_path"])] = x

print("Precomputing text embeddings (cross-attention fusion)...")
ret_db = []

# detecteaza disease category din prompt (incepe cu numele bolii)
DISEASE_MAP = {
    "age-related": "AMD",
    "diabetic":    "DME",
    "drusen":      "DRUSEN",
    "normal":      "NORMAL",
}

def detect_disease(prompt_a):
    p = prompt_a.lower()
    for kw, cls in DISEASE_MAP.items():
        if p.startswith(kw):
            return cls
    # fallback: cauta in text
    if "macular degeneration" in p or "amd" in p:
        return "AMD"
    if "diabetic" in p or "dme" in p:
        return "DME"
    if "drusen" in p:
        return "DRUSEN"
    if "normal" in p:
        return "NORMAL"
    return "UNKNOWN"

for img_path, item in split_raw.items():
    pa = item.get("a", "")
    pb = item.get("b", "")
    if not pa or not pb:
        continue

    ta = proc.tokenizer(pa, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    tb = proc.tokenizer(pb, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    with torch.no_grad():
        ma = ta.get("attention_mask", torch.ones_like(ta["input_ids"]))
        mb = tb.get("attention_mask", torch.ones_like(tb["input_ids"]))
        ea = model.encode_text(ta["input_ids"].to(DEV), ma.to(DEV))
        eb = model.encode_text(tb["input_ids"].to(DEV), mb.to(DEV))
        merged = model.fusion(ea, eb)

    path_norm = norm_path(img_path)
    sev_info  = sev_lookup.get(path_norm, {})
    disease   = detect_disease(pa)

    ret_db.append({
        "emb":      merged.cpu(),
        "prompt_a": pa,
        "prompt_b": pb,
        "disease":  disease,
        "path":     img_path,
        "sev":      sev_info.get("severity_percent"),
        "sev_level": sev_info.get("severity_level"),
    })

all_txt_emb = torch.cat([r["emb"] for r in ret_db])
print(f"Retrieval DB: {len(ret_db)} entries")
print("Ready!\n")


# ---------- inference ----------

def get_sev_level(pct):
    if pct < 15:
        return "Minimal"
    if pct < 30:
        return "Mild"
    if pct < 50:
        return "Moderate"
    if pct < 70:
        return "Significant"
    if pct < 85:
        return "Severe"
    return "Critical"


def analyze(image):
    if image is None:
        return None, "Upload an OCT image to analyze."

    pil = Image.fromarray(image).convert("RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil = auto_crop(pil)

    inputs = proc(images=pil, return_tensors="pt")
    pv = inputs["pixel_values"].to(DEV)

    with torch.no_grad():
        ie      = model.encode_image(pv)
        logits  = model.cls_head(ie)
        sev_pct = model.sev_head(ie).clamp(0, 1).item() * 100

    probs    = torch.softmax(logits, dim=1)[0]
    pred_i   = probs.argmax().item()
    pred_cls = classes[pred_i]
    conf     = probs[pred_i].item() * 100
    per_cls  = {classes[i]: float(probs[i]) for i in range(len(classes))}

    # EigenCAM
    rgb_resized = cv2.resize(np.array(pil), (448, 448))
    rgb_float   = np.float32(rgb_resized) / 255.0

    grayscale_cam = cam_gen(input_tensor=pv, targets=None)[0, :]
    grayscale_cam = smooth_cam(grayscale_cam)

    gray_img    = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
    retina_mask = (gray_img > 35).astype(np.float32)
    retina_mask = cv2.GaussianBlur(retina_mask, (15, 15), 0)
    grayscale_cam = grayscale_cam * retina_mask

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # retrieval top 3
    with torch.no_grad():
        sim = (ie.cpu() @ all_txt_emb.T).squeeze(0)
    top = sim.topk(3)

    matches = ""
    for rank, (score, idx) in enumerate(zip(top.values, top.indices)):
        r = ret_db[idx.item()]
        matches += f"\n{'─' * 50}\n"
        matches += f"Match #{rank + 1} (similarity: {score.item():.3f})\n"
        matches += f"Disease: {r['disease']}"
        if r["sev"] is not None:
            matches += f" | Severity: {r['sev']:.0f}%"
        matches += f"\n\nStructure:\n{r['prompt_a']}\n"
        matches += f"\nLesions:\n{r['prompt_b']}\n"

    level   = get_sev_level(sev_pct)
    bar_sev = int(sev_pct / 2)

    report = f"""
DIAGNOSIS: {pred_cls}
Confidence: {conf:.1f}%

SEVERITY: {sev_pct:.1f}% ({level})
{'█' * bar_sev}{'░' * (50 - bar_sev)}

CONFIDENCE PER CLASS:
  AMD:    {'█' * int(per_cls.get('AMD', 0) * 50)} {per_cls.get('AMD', 0) * 100:.1f}%
  DME:    {'█' * int(per_cls.get('DME', 0) * 50)} {per_cls.get('DME', 0) * 100:.1f}%
  DRUSEN: {'█' * int(per_cls.get('DRUSEN', 0) * 50)} {per_cls.get('DRUSEN', 0) * 100:.1f}%
  NORMAL: {'█' * int(per_cls.get('NORMAL', 0) * 50)} {per_cls.get('NORMAL', 0) * 100:.1f}%

SIMILAR CASES FROM DATABASE:
{matches}
"""

    return overlay, report


# ---------- gradio app ----------

def main():
    import gradio as gr

    with gr.Blocks(
        title="MedSigLIP v7 OCT Analyzer",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
        # MedSigLIP v7 — Retinal OCT Analyzer
        ### MedGemma 27B Prompts + Cross-Attention Fusion + Dual Contrastive + Severity Estimation

        Upload a retinal OCT scan to get:
        - **Disease Classification** (AMD, DME, DRUSEN, NORMAL)
        - **Severity Estimation** (0-100%)
        - **EigenCAM Heatmap** (where the model looks)
        - **Similar Cases** from the database with detailed descriptions
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="Upload OCT Scan", type="numpy")
                btn     = gr.Button("Analyze", variant="primary", size="lg")
                gr.Markdown("Upload any retinal OCT B-scan image (grayscale or RGB)")

            with gr.Column(scale=1):
                out_img = gr.Image(label="EigenCAM Attention Map", type="numpy")

        with gr.Row():
            out_report = gr.Textbox(label="Analysis Report", lines=30, max_lines=50)

        btn.click(fn=analyze, inputs=[inp_img], outputs=[out_img, out_report])

        gr.Markdown("""
        ---
        **Thesis Project** — Retinal OCT Disease Classification using MedSigLIP Multi-Task Learning

        *Pipeline: MedGemma 27B IT → Gemini Flash-Lite (split) → MedSigLIP v7 (cross-attention + dual contrastive)*

        Explainability: EigenCAM (layers[-2], blur=31, threshold=35%)

        This is a research tool, not a medical diagnostic device.
        """)

    app.launch(share=False, server_name="localhost", server_port=7860)


if __name__ == "__main__":
    main()