"""
MedSigLIP Question-Answering Demo

Model de tip question-answer prin retrieval contrastiv:
  1. Imagine OCT → encode_image → img_emb
  2. Intrebare text → encode_text → q_emb
  3. Retrieval: gaseste descrierile din dataset cele mai relevante
     pentru AMBELE (imaginea SI intrebarea)
  4. Descrierile gasite = RASPUNSUL

Plus informatii din heads:
  - cls_head → boala clasificata
  - sev_head → severitate estimata
  - biomarker heads v3 → biomarkeri detectati

Rulare:
    python -m src.demo.qa_demo --image "path/to/oct.png"
    python -m src.demo.qa_demo --image "path/to/oct.png" --question "Are there drusen?"
    python -m src.demo.qa_demo --interactive

"""

import argparse
import json
import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset, collate_oct5k
from src.utils.seed import set_seed


# ================================================================
# CONFIG
# ================================================================

class Config:
    model_path  = "models/medsiglip-448"
    ckpt_path   = "experiments/medsiglip_v5/ckpts/best.pth"
    bm_ckpt     = "experiments/biomarker_heads_v3/ckpts/best.pth"

    # knowledge base — tot dataset-ul
    splits_dir  = "data/oct5k/splits"
    split_json  = "data/oct5k/medgemma_prompts_split.json"
    sev_json    = "data/oct5k/severity_scores_combined.json"

    # retrieval
    top_k       = 5
    alpha       = 0.7  # weight imagine vs intrebare (0.7 = 70% imagine, 30% intrebare)

    bs      = 8
    workers = 0
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    amp     = torch.cuda.is_available()


cfg = Config()

CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]


# ================================================================
# MODELS
# ================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn_oct2mask = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_mask2oct = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, emb_a, emb_b):
        a, b = emb_a.unsqueeze(1), emb_b.unsqueeze(1)
        attn_a, _ = self.attn_oct2mask(query=a, key=b, value=b)
        attn_b, _ = self.attn_mask2oct(query=b, key=a, value=a)
        attn_a, attn_b = attn_a.squeeze(1), attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


class MedSigLIPMultiTask(nn.Module):
    def __init__(self, model_path, n_classes=4):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))
        dim = self.backbone.config.vision_config.hidden_size
        self.sev_head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.cls_head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_classes))
        self.fusion   = CrossAttentionFusion(dim, heads=4, dropout=0.1)

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


class BiomarkerHeadsV3(nn.Module):
    def __init__(self, backbone, dim, n_bm=9, unfreeze_last_n=2):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        vision_layers = self.backbone.vision_model.encoder.layers
        n_layers = len(vision_layers)
        for i in range(max(0, n_layers - unfreeze_last_n), n_layers):
            for p in vision_layers[i].parameters():
                p.requires_grad = True
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            ) for _ in range(n_bm)
        ])

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values):
        img_emb = self.encode_image(pixel_values)
        logits = torch.cat([h(img_emb) for h in self.heads], dim=-1)
        return logits


# ================================================================
# KNOWLEDGE BASE — precompute embeddings din dataset
# ================================================================

class KnowledgeBase:
    """Precomputed embeddings + descriptions din tot dataset-ul."""

    def __init__(self):
        self.txt_embs   = None   # [N, dim]
        self.img_embs   = None   # [N, dim]
        self.diseases   = []     # [N]
        self.paths      = []     # [N]
        self.prompts_a  = []     # [N] structura
        self.prompts_b  = []     # [N] patologie
        self.n = 0

    @torch.no_grad()
    def build(self, model, dataset, loader, proc):
        print("  Building knowledge base...")
        model.eval()

        all_img, all_txt = [], []

        for batch_idx, batch in enumerate(tqdm(loader, desc="  KB embeddings")):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
            ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
            ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
            mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

            with autocast(cfg.device, enabled=cfg.amp):
                ie = model.encode_image(pv)
                ea = model.encode_text(ia, ma)
                eb = model.encode_text(ib, mb)
                merged = model.fusion(ea, eb)

            all_img.append(ie.cpu())
            all_txt.append(merged.cpu())

            start = batch_idx * cfg.bs
            end   = min(start + cfg.bs, len(dataset))
            for i in range(end - start):
                row = dataset.df.iloc[start + i]
                self.diseases.append(row["disease"])
                self.paths.append(row["image_path"])

                prompts = dataset.prompts.get(row["image_path"], {})
                self.prompts_a.append(prompts.get("a", ""))
                self.prompts_b.append(prompts.get("b", ""))

            del pv, ia, ma, ib, mb, ie, ea, eb, merged

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.img_embs = torch.cat(all_img)
        self.txt_embs = torch.cat(all_txt)
        self.n = len(self.diseases)
        print(f"  Knowledge base: {self.n} entries")


# ================================================================
# QA ENGINE
# ================================================================

class QAEngine:

    def __init__(self, model, bm_model, bm_thresholds, proc, kb):
        self.model         = model
        self.bm_model      = bm_model
        self.bm_thresholds = bm_thresholds
        self.proc          = proc
        self.kb            = kb

    @torch.no_grad()
    def answer(self, image_path, question=None):
        """
        Raspunde la o intrebare despre o imagine OCT.

        Args:
            image_path: calea catre imaginea OCT
            question: intrebarea (optional — daca lipseste, returneaza diagnostic complet)

        Returns:
            dict cu disease, severity, biomarkers, retrieved descriptions
        """
        self.model.eval()

        # --- encode imagine ---
        img = Image.open(image_path).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        px  = self.proc(images=img, return_tensors="pt")
        pv  = px["pixel_values"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            img_emb = self.model.encode_image(pv)

            # clasificare
            cls_logits = self.model.cls_head(img_emb)
            cls_probs  = torch.softmax(cls_logits, dim=1)[0]
            cls_idx    = cls_probs.argmax().item()
            cls_conf   = cls_probs[cls_idx].item()

            # severitate
            sev = self.model.sev_head(img_emb).clamp(0, 1).item() * 100

        # --- biomarkeri v3 ---
        detected_bm = []
        if self.bm_model is not None:
            with autocast(cfg.device, enabled=cfg.amp):
                bm_logits = self.bm_model(pv)
                bm_probs  = torch.sigmoid(bm_logits)[0].cpu().numpy()

            for i, bm in enumerate(BIOMARKERS):
                if bm_probs[i] > self.bm_thresholds[i]:
                    detected_bm.append({
                        "name": bm,
                        "confidence": round(float(bm_probs[i]) * 100, 1),
                    })

        # --- retrieval ---
        img_emb_cpu = img_emb.cpu()

        # similarity cu toate descrierile din KB
        img_scores = (img_emb_cpu @ self.kb.txt_embs.T).squeeze(0)

        if question:
            # encode intrebarea
            tok = self.proc.tokenizer(
                question, padding="max_length", truncation=True,
                max_length=64, return_tensors="pt",
            )
            ids  = tok["input_ids"].to(cfg.device)
            mask = tok.get("attention_mask", torch.ones_like(ids)).to(cfg.device)

            with autocast(cfg.device, enabled=cfg.amp):
                q_emb = self.model.encode_text(ids, mask)

            q_emb_cpu = q_emb.cpu()
            q_scores  = (q_emb_cpu @ self.kb.txt_embs.T).squeeze(0)

            # combined score: alpha * imagine + (1-alpha) * intrebare
            combined = cfg.alpha * img_scores + (1 - cfg.alpha) * q_scores
        else:
            combined = img_scores

        # top-K retrieval
        top_vals, top_ids = combined.topk(cfg.top_k)
        retrieved = []
        for rank, (score, idx) in enumerate(zip(top_vals, top_ids)):
            idx = idx.item()
            retrieved.append({
                "rank":     rank + 1,
                "disease":  self.kb.diseases[idx],
                "score":    round(score.item(), 4),
                "structure": self.kb.prompts_a[idx],
                "pathology": self.kb.prompts_b[idx],
            })

        del pv, img_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "image_path":  image_path,
            "question":    question,
            "disease":     CLASSES[cls_idx],
            "confidence":  round(cls_conf * 100, 1),
            "severity":    round(sev, 1),
            "biomarkers":  detected_bm,
            "retrieved":   retrieved,
        }


# ================================================================
# DISPLAY
# ================================================================

def display_answer(result):
    print(f"\n{'=' * 70}")
    print(f"  MedSigLIP Question-Answering")
    print(f"{'=' * 70}")
    print(f"  Image: {result['image_path']}")

    if result["question"]:
        print(f"  Question: {result['question']}")

    print(f"\n  --- Diagnostic ---")
    print(f"  Disease:    {result['disease']} ({result['confidence']}% confidence)")
    print(f"  Severity:   {result['severity']}%")

    if result["biomarkers"]:
        print(f"\n  --- Biomarkers Detected ---")
        for bm in result["biomarkers"]:
            print(f"    {bm['name']:<25} {bm['confidence']}%")
    else:
        print(f"\n  No biomarkers detected above threshold.")

    print(f"\n  --- Retrieved Descriptions (top {len(result['retrieved'])}) ---")
    for r in result["retrieved"]:
        print(f"\n  #{r['rank']} [{r['disease']}] (score={r['score']:.4f})")
        print(f"    Structure: {r['structure'][:100]}...")
        print(f"    Pathology: {r['pathology'][:100]}...")

    print(f"\n{'=' * 70}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="MedSigLIP QA Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to OCT image")
    parser.add_argument("--question", type=str, default=None, help="Question about the image")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    set_seed()

    print("=" * 70)
    print("  MedSigLIP Question-Answering System")
    print("  Retrieval-based QA: image + question -> relevant descriptions")
    print("=" * 70)

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    # --- load MedSigLIP ---
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model = MedSigLIPMultiTask(cfg.model_path, n_classes=4)
    state_dict = {}
    for k, v in ckpt["model"].items():
        if "fusion.attn_a2b" in k:
            k = k.replace("fusion.attn_a2b", "fusion.attn_oct2mask")
        elif "fusion.attn_b2a" in k:
            k = k.replace("fusion.attn_b2a", "fusion.attn_mask2oct")
        state_dict[k] = v
    model.load_state_dict(state_dict)
    model = model.to(cfg.device)
    model.eval()
    print(f"  MedSigLIP: loaded")

    # --- load biomarker heads v3 ---
    bm_model = None
    bm_thresholds = [0.5] * len(BIOMARKERS)

    if os.path.exists(cfg.bm_ckpt):
        bm_ckpt_data = torch.load(cfg.bm_ckpt, map_location="cpu", weights_only=False)
        backbone_v3 = AutoModel.from_pretrained(cfg.model_path, torch_dtype=torch.float32)
        bb_state = {
            k.replace("backbone.", ""): v
            for k, v in ckpt["model"].items()
            if k.startswith("backbone.")
        }
        backbone_v3.load_state_dict(bb_state, strict=True)

        dim = backbone_v3.config.vision_config.hidden_size
        bm_model = BiomarkerHeadsV3(backbone_v3, dim, n_bm=len(BIOMARKERS), unfreeze_last_n=2)
        bm_model.load_state_dict(bm_ckpt_data["model"])
        bm_model = bm_model.to(cfg.device)
        bm_model.eval()
        bm_thresholds = [float(t) for t in bm_ckpt_data.get("thresholds", [0.5] * len(BIOMARKERS))]
        print(f"  Biomarker v3: loaded")

    # --- build knowledge base ---
    import pandas as pd
    train_df = pd.read_csv(f"{cfg.splits_dir}/train.csv")
    val_df   = pd.read_csv(f"{cfg.splits_dir}/val.csv")
    test_df  = pd.read_csv(f"{cfg.splits_dir}/test.csv")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_csv  = f"{cfg.splits_dir}/_tmp_qa_all.csv"
    all_df.to_csv(all_csv, index=False)

    dataset = OCT5kDataset(
        split_csv=all_csv, split_json=cfg.split_json,
        severity_json=cfg.sev_json, processor=proc, mode="eval",
    )
    loader = DataLoader(
        dataset, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, collate_fn=collate_oct5k,
    )

    kb = KnowledgeBase()
    kb.build(model, dataset, loader, proc)

    # cleanup temp
    if os.path.exists(all_csv):
        os.remove(all_csv)

    # --- QA Engine ---
    qa = QAEngine(model, bm_model, bm_thresholds, proc, kb)
    print(f"\n  QA System ready! ({kb.n} entries in knowledge base)\n")

    # --- run ---
    if args.interactive:
        print("  Interactive mode. Type 'quit' to exit.\n")
        while True:
            image_path = input("  Image path: ").strip()
            if image_path.lower() == "quit":
                break
            if not os.path.exists(image_path):
                print(f"  File not found: {image_path}")
                continue

            question = input("  Question (Enter for full diagnostic): ").strip()
            if not question:
                question = None

            result = qa.answer(image_path, question)
            display_answer(result)

    elif args.image:
        if not os.path.exists(args.image):
            print(f"  File not found: {args.image}")
            return

        result = qa.answer(args.image, args.question)
        display_answer(result)

        # save result
        out_path = "experiments/qa_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

    else:
        # demo cu o imagine random din test set
        import random
        random.seed(42)
        test_df = pd.read_csv(f"{cfg.splits_dir}/test.csv")
        row = test_df.sample(1).iloc[0]
        img_path = dataset._locate(row["image_path"])

        if img_path is None:
            print("  Could not find demo image")
            return

        print(f"  Demo with random test image: {row['disease']}")

        # Q1: diagnostic complet
        r1 = qa.answer(img_path)
        display_answer(r1)

        # Q2: intrebare specifica
        questions = [
            "Are there any drusen deposits present?",
            "Is there fluid in the retinal layers?",
            "What is the severity of this condition?",
            "Is the photoreceptor layer intact?",
        ]
        for q in questions:
            r = qa.answer(img_path, q)
            print(f"\n  Q: {q}")
            print(f"  A: Top match [{r['retrieved'][0]['disease']}]: {r['retrieved'][0]['pathology'][:120]}...")
            print(f"     Score: {r['retrieved'][0]['score']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  QA Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()