"""
Gradio Demo: MedSigLIP Multi-Task OCT Analyzer

Interfata web unde uploadezi o imagine OCT si primesti:
  1. Clasificare boala (AMD/DME/DRUSEN/NORMAL) cu confidenta
  2. Severity estimation (0-100%)
  3. GradCAM heatmap (unde se uita modelul)
  4. Top 3 retrieval - cele mai similare descrieri din dataset

Rulare:
    python -m src.demo.gradio_app
    sau: python gradio_app.py

Se deschide automat in browser la http://localhost:7860
"""

import os
import sys
import json
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

MODEL_PATH = "models/medsiglip-448"
CHECKPOINT = "experiments/medsiglip_pipeline/ckpts/best.pth"
SPLIT_JSON = "data/oct5k/medgemma_prompts_split.json"
SEVERITY_JSON = "data/oct5k/severity_scores.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]


# ═══════════════════════════════════════════════════════════════════════
# MODEL
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

    def encode_image(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# GRADCAM
# ═══════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.model.vision_model.encoder.layers[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
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
        self.model.zero_grad()

        img_features = self.model.model.get_image_features(pixel_values=pixel_values)
        if hasattr(img_features, "pooler_output"):
            img_emb = img_features.pooler_output
        elif hasattr(img_features, "last_hidden_state"):
            img_emb = img_features.last_hidden_state[:, 0]
        else:
            img_emb = img_features

        cls_logits = self.model.cls_head(img_emb)

        if target_class is None:
            target_class = cls_logits.argmax(dim=1)

        one_hot = torch.zeros_like(cls_logits)
        for i in range(len(target_class)):
            one_hot[i, target_class[i]] = 1

        cls_logits.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[:, 1:, :]
        activations = self.activations[:, 1:, :]

        weights = gradients.mean(dim=-1, keepdim=True)
        cam = (weights * activations).sum(dim=-1)
        cam = F.relu(cam)

        num_patches = cam.shape[1]
        h = w = int(num_patches ** 0.5)
        cam = cam[:, :h * w]
        cam = cam.view(-1, h, w)

        for i in range(cam.shape[0]):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max - cam_min > 0:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()


def make_overlay(image_np, heatmap, alpha=0.5):
    h, w = image_np.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]

    overlay = (1 - alpha) * image_np.astype(np.float32) / 255.0 + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL + RETRIEVAL DATABASE
# ═══════════════════════════════════════════════════════════════════════

print("Loading MedSigLIP model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
nc = ckpt.get("num_classes", 4)
classes = ckpt.get("classes", CLASSES)

model = MedSigLIPMultiTask(MODEL_PATH, num_classes=nc)
model.load_state_dict(ckpt["model"])
model = model.to(DEVICE)
model.eval()

gradcam = GradCAM(model)

# incarcam retrieval database (prompturi split)
print("Loading retrieval database...")
with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(SEVERITY_JSON, "r", encoding="utf-8") as f:
    sev_data = json.load(f)

# dict pt severity lookup
sev_dict = {item["image_path"]: item for item in sev_data if item.get("severity_valid")}

# precompute text embeddings pt retrieval
print("Precomputing text embeddings for retrieval...")
retrieval_db = []

for item in split_data:
    if not item.get("split_valid"):
        continue

    prompt_a = item["prompt_a"]
    prompt_b = item["prompt_b"]

    # tokenize ambele
    tok_a = processor.tokenizer(prompt_a, padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt")
    tok_b = processor.tokenizer(prompt_b, padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt")

    with torch.no_grad():
        mask_a = tok_a.get("attention_mask", torch.ones_like(tok_a["input_ids"]))
        mask_b = tok_b.get("attention_mask", torch.ones_like(tok_b["input_ids"]))
        emb_a = model.encode_text(tok_a["input_ids"].to(DEVICE), mask_a.to(DEVICE))
        emb_b = model.encode_text(tok_b["input_ids"].to(DEVICE), mask_b.to(DEVICE))
        merged = F.normalize((emb_a + emb_b) / 2, p=2, dim=-1)

    sev_info = sev_dict.get(item["image_path"], {})

    retrieval_db.append({
        "embedding": merged.cpu(),
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "disease": item["disease_category"],
        "image_path": item["image_path"],
        "severity": sev_info.get("severity_percent", None),
        "severity_level": sev_info.get("severity_level", None),
    })

# stack all embeddings
all_text_embs = torch.cat([r["embedding"] for r in retrieval_db])
print(f"Retrieval DB: {len(retrieval_db)} entries loaded!")
print("Ready!\n")


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def analyze_oct(image):
    if image is None:
        return None, "Upload an OCT image to analyze."

    # preprocess
    image_pil = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image_pil, return_tensors="pt")
    pv = inputs["pixel_values"].to(DEVICE)

    # classification + severity
    with torch.no_grad():
        img_emb = model.encode_image(pv)
        cls_logits = model.cls_head(img_emb)
        sev_pred = model.severity_head(img_emb).item() * 100

    probs = torch.softmax(cls_logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx].item() * 100

    # confidence per class
    conf_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

    # GradCAM
    heatmap = gradcam.generate(pv)
    overlay = make_overlay(image, heatmap[0])

    # retrieval: top 3 cele mai similare
    with torch.no_grad():
        sim = (img_emb.cpu() @ all_text_embs.T).squeeze(0)
    top_k = sim.topk(3)

    retrieval_text = ""
    for rank, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
        r = retrieval_db[idx.item()]
        retrieval_text += f"\n{'─' * 50}\n"
        retrieval_text += f"Match #{rank + 1} (similarity: {score.item():.3f})\n"
        retrieval_text += f"Disease: {r['disease']}"
        if r['severity'] is not None:
            retrieval_text += f" | Severity: {r['severity']:.0f}%"
        retrieval_text += f"\n\nStructure:\n{r['prompt_a']}\n"
        retrieval_text += f"\nLesions:\n{r['prompt_b']}\n"

    # severity level
    if sev_pred < 15:
        sev_level = "Minimal"
    elif sev_pred < 30:
        sev_level = "Mild"
    elif sev_pred < 50:
        sev_level = "Moderate"
    elif sev_pred < 70:
        sev_level = "Significant"
    elif sev_pred < 85:
        sev_level = "Severe"
    else:
        sev_level = "Critical"

    # raport complet
    report = f"""
╔══════════════════════════════════════════════╗
║         MedSigLIP OCT Analysis Report        ║
╚══════════════════════════════════════════════╝

DIAGNOSIS: {pred_class}
Confidence: {confidence:.1f}%

SEVERITY: {sev_pred:.1f}% ({sev_level})
{'█' * int(sev_pred / 2)}{'░' * (50 - int(sev_pred / 2))}

CONFIDENCE PER CLASS:
  AMD:    {'█' * int(conf_dict['AMD'] * 50)} {conf_dict['AMD']*100:.1f}%
  DME:    {'█' * int(conf_dict['DME'] * 50)} {conf_dict['DME']*100:.1f}%
  DRUSEN: {'█' * int(conf_dict['DRUSEN'] * 50)} {conf_dict['DRUSEN']*100:.1f}%
  NORMAL: {'█' * int(conf_dict['NORMAL'] * 50)} {conf_dict['NORMAL']*100:.1f}%

SIMILAR CASES FROM DATABASE:
{retrieval_text}
"""

    return overlay, report


# ═══════════════════════════════════════════════════════════════════════
# GRADIO APP
# ═══════════════════════════════════════════════════════════════════════

def main():
    import gradio as gr

    with gr.Blocks(
        title="MedSigLIP OCT Analyzer",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
        # 🔬 MedSigLIP — Retinal OCT Analyzer
        ### Multi-Task Deep Learning for Retinal Disease Classification, Severity Estimation & Visual Explanation

        Upload a retinal OCT scan to get:
        - **Disease Classification** (AMD, DME, DRUSEN, NORMAL)
        - **Severity Estimation** (0-100%)
        - **GradCAM Heatmap** (where the model looks)
        - **Similar Cases** from the database with detailed descriptions
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload OCT Scan", type="numpy")
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

                gr.Markdown("### Example Images")
                gr.Markdown("Upload any retinal OCT B-scan image (grayscale or RGB)")

            with gr.Column(scale=1):
                output_image = gr.Image(label="GradCAM Attention Map", type="numpy")

        with gr.Row():
            output_report = gr.Textbox(
                label="Analysis Report",
                lines=30,
                max_lines=50,
            )

        analyze_btn.click(
            fn=analyze_oct,
            inputs=[input_image],
            outputs=[output_image, output_report],
        )

        gr.Markdown("""
        ---
        **Thesis Project** — Retinal OCT Disease Classification using MedSigLIP Multi-Task Learning

        *Pipeline: MedGemma (prompt generation) → Gemini Flash (prompt splitting) → Qwen2.5 (severity) → MedSigLIP (fine-tuning)*

        ⚠️ This is a research tool, not a medical diagnostic device.
        """)

    app.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()