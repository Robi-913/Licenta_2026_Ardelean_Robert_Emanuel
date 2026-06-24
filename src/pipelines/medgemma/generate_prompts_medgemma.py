"""
Step 2: Generate Medical Descriptive Captions cu MedGemma 27B IT

Schimbari fata de v1:
  - MedGemma 4B → 27B IT (multimodal, mai detaliat)
  - Cuantificare 4-bit pentru 24GB VRAM
  - metadata_v2 cu bbox_source: doctor / yolo / none
  - Prompt adaptat per bbox_source:
      doctor → descriere completa cu bbox corelate cu straturi
      yolo   → descriere cu mentiune ca bbox sunt silver labels (YOLO)
      none   → descriere generala din imagine si grosimi straturi
  - Output: medgemma_prompts_v2.json

  LOW MEMORY FIX (32GB RAM + 24GB VRAM):
  - torch_dtype=bfloat16 (era None → float32 peak la load = dublu RAM)
  - max_memory explicit pt GPU + CPU
  - offload_folder pe disk ca safety net
  - bnb_4bit_quant_type="nf4" explicit
  - gc agresiv inainte de load

Rulare:
    python src/pipelines/medgemma/build_text_prompt.py
"""

import gc
import json
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.utils.seed import set_seed


# ---------- config ----------

class Config:
    model_path = "models/medgemma-27b-it"
    master_json    = "data/oct5k/metadata_v2/_master.json"
    output_json = "data/oct5k/medgemma_prompts_v2_27b.json"

    max_tokens     = 256
    save_interval  = 50
    resume         = True
    device         = "cuda" if torch.cuda.is_available() else "cpu"

    load_in_4bit   = True

    # --- LOW MEMORY SETTINGS ---
    # 32GB RAM total, ~3-4GB pt sistem => ~28GB utilizabil
    max_cpu_mem    = "26GiB"
    # 24GB VRAM, lasam 2GB pt KV cache si overhead
    max_gpu_mem    = "22GiB"
    # folder pt disk offload daca nici CPU RAM nu incape
    offload_dir    = "offload_tmp"


cfg = Config()


# ---------- helpers ----------

def save_results(data):
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_model():
    print(f"\n  Model: {cfg.model_path}")
    print(f"  4-bit quantization: {cfg.load_in_4bit}")
    print(f"  Max GPU mem: {cfg.max_gpu_mem}, Max CPU mem: {cfg.max_cpu_mem}")

    # --- curatam tot inainte de load ---
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    quant_cfg = None
    if cfg.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",          # explicit nf4
        )

    # limita memorie: accelerate pune layere pe disk daca depaseste
    max_mem = {0: cfg.max_gpu_mem, "cpu": cfg.max_cpu_mem}

    os.makedirs(cfg.offload_dir, exist_ok=True)

    proc = AutoProcessor.from_pretrained(cfg.model_path)

    # CRUCIAL: torch_dtype=bfloat16, NU None
    # cu None, shardurile se incarca la float32 inainte de quantizare
    # asta inseamna peak RAM ~54GB pt 27B, imposibil pe 32GB
    # cu bfloat16, peak RAM ~27GB, incape (tight dar merge)
    mdl = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        quantization_config=quant_cfg,
        device_map="auto",
        max_memory=max_mem,
        offload_folder=cfg.offload_dir,     # safety net: disk offload
        low_cpu_mem_usage=True,             # shard-by-shard loading
        torch_dtype=torch.bfloat16,         # FIX: era None → peak dublu
    )
    mdl.eval()

    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024 ** 3
        peak  = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM: {used:.1f}GB used / {peak:.1f}GB peak / {total:.1f}GB total")

    # print device map pt debug
    if hasattr(mdl, "hf_device_map"):
        devs = Counter(str(v) for v in mdl.hf_device_map.values())
        print(f"  Device map: {dict(devs)}")

    print("  Loaded!\n")
    return mdl, proc


# ---------- prompt builder ----------

def build_prompt(meta):
    disease     = str(meta.get("disease_category", "UNKNOWN")).upper()
    bbox_source = meta.get("bbox_source", "none")   # doctor | yolo | none

    # --- header adaptat per bbox_source ---
    if bbox_source == "doctor":
        header = (
            f"You are an expert ophthalmic image analyzer. "
            f"Provide a highly detailed, objective morphological description of this retinal OCT scan ({disease}). "
            f"The following lesion annotations were manually labeled by medical doctors and are ground truth."
        )
    elif bbox_source == "yolo":
        header = (
            f"You are an expert ophthalmic image analyzer. "
            f"Provide a highly detailed, objective morphological description of this retinal OCT scan ({disease}). "
            f"The following lesion annotations were automatically detected by a YOLO object detection model "
            f"and may contain imprecisions. Use them as supporting context, not as ground truth."
        )
    else:
        header = (
            f"You are an expert ophthalmic image analyzer. "
            f"Provide a highly detailed, objective morphological description of this retinal OCT scan ({disease}). "
            f"No lesion annotations are available — describe what you observe directly in the image."
        )

    parts = [
        header,
        "Focus STRICTLY on describing the visual features, layer geometries, thickness, structural deformations, and lesions.",
        "Do NOT compute severity. Do NOT make clinical diagnoses or treatment suggestions. Just describe what is visible.",
    ]

    # --- boundaries ---
    has_bounds = meta.get("has_boundaries") and meta.get("boundaries")
    if has_bounds:
        b = meta["boundaries"]

        trt = b.get("total_retinal_thickness", {})
        if trt:
            parts.append(
                f"- Total Retinal Thickness: mean {trt.get('mean_px', 0):.1f}px "
                f"(range {trt.get('min_px', 0)}-{trt.get('max_px', 0)}px)."
            )

        regs = b.get("regions", {})
        if regs:
            items = [f"{name} ({d.get('mean_thickness_px', 0):.1f}px)" for name, d in regs.items()]
            parts.append("- Layer Thicknesses: " + ", ".join(items) + ".")

        n_def = int(b.get("num_deformations", 0))
        if n_def > 0:
            zones    = b.get("deformation_zones", [])
            zone_set = list(set(d.get("zone", "unknown") for d in zones))
            type_set = list(set(d.get("type", "deformation") for d in zones))
            max_dev  = max((d.get("deviation_from_mean_px", 0) for d in zones), default=0)

            parts.append(
                f"- Structural Deformations: {n_def} abnormal points. "
                f"Types: {', '.join(type_set)}. Zones: {', '.join(zone_set)}. "
                f"Max deviation: {max_dev:+.1f}px."
            )

    # --- lesions ---
    n_les   = int(meta.get("num_lesions", 0))
    has_les = meta.get("has_bounding_boxes") and n_les > 0

    if has_les:
        cls_list = ", ".join(sorted(set(meta.get("lesion_classes", []))))
        area     = meta.get("total_lesion_area_percent", 0)
        src_note = " (YOLO-detected, confidence shown)" if bbox_source == "yolo" else " (doctor-annotated)"

        parts.append(
            f"- Lesions{src_note}: {n_les} lesions ({cls_list}), "
            f"covering {area:.1f}% of the image area."
        )

        for les in meta.get("lesions", [])[:5]:
            layer = les.get("layer_correlation", {}).get("affected_layer", "unknown")
            zone  = les.get("retinal_zone", "unknown")
            sz    = les.get("size_px", [0, 0])

            # pentru YOLO: adauga confidence
            conf_note = ""
            if bbox_source == "yolo":
                conf = les.get("yolo_confidence", 0)
                conf_note = f", conf={conf:.2f}"

            parts.append(
                f"  * {les.get('class', 'Lesion')} at {zone} "
                f"(Layer: {layer}), size: {sz[0]}x{sz[1]}px{conf_note}."
            )
    else:
        parts.append(
            "- Lesions: No lesion annotations available. "
            "Describe any visible structural abnormalities based on the image."
        )

    parts.append(
        "\nInstruction: Write a single, fluent, descriptive paragraph integrating "
        "these facts with what you observe in the image. "
        "Do not use bullet points. "
        "Start directly with the description of the retinal structure."
    )

    return "\n".join(parts)


# ---------- generation ----------

@torch.no_grad()
def run_generate(mdl, proc, msgs, input_keys):
    inputs = proc.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    prefix_len = inputs["input_ids"].shape[1]

    feed = {k: inputs[k].to(mdl.device) for k in input_keys if k in inputs}

    out = mdl.generate(
        **feed,
        max_new_tokens=cfg.max_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=proc.tokenizer.eos_token_id,
    )

    decoded = proc.decode(out[0][prefix_len:], skip_special_tokens=True)

    # eliberam tensori dupa fiecare generare
    del inputs, feed, out
    return decoded.strip()


# ---------- image locator ----------

IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


def locate_image(meta):
    """Returneaza imaginea OCT originala (nu masca)."""
    disk = meta.get("image_disk_path", "")
    if disk and Path(disk).exists():
        return str(disk)
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in IMG_DIRS:
        full = Path(base) / rel
        if full.exists():
            return str(full)
        for ext in [".png", ".jpeg", ".jpg"]:
            alt = full.with_suffix(ext)
            if alt.exists():
                return str(alt)
    return None


# ---------- main loop ----------

def process_all(mdl, proc, metadata):
    done = {}
    if cfg.resume and Path(cfg.output_json).exists():
        try:
            with open(cfg.output_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                done = {
                    x["image_path"]: x["generated_prompt"]
                    for x in prev
                    if isinstance(x, dict)
                    and "image_path" in x
                    and "generated_prompt" in x
                }
        except Exception as e:
            print(f"  WARNING resume: {e}")
        print(f"  Resume: {len(done)} already done")

    results = []
    n_skip  = 0
    n_err   = 0

    for i, meta in enumerate(tqdm(metadata, desc="  Generating")):
        img_path    = meta["image_path"]
        disease     = meta["disease_category"]
        bbox_source = meta.get("bbox_source", "none")

        if img_path in done:
            results.append({
                "image_path":       img_path,
                "disease_category": disease,
                "bbox_source":      bbox_source,
                "generated_prompt": done[img_path],
            })
            n_skip += 1
            continue

        txt      = build_prompt(meta)
        img_file = locate_image(meta)

        try:
            content = []

            if img_file and os.path.exists(img_file):
                with Image.open(img_file) as im:
                    content.append({"type": "image", "image": im.convert("RGB").copy()})

            content.append({"type": "text", "text": txt})

            has_img = any(c["type"] == "image" for c in content)
            keys    = ["input_ids", "attention_mask"]
            if has_img:
                keys.append("pixel_values")

            msgs    = [{"role": "user", "content": content}]
            caption = run_generate(mdl, proc, msgs, keys)

        except Exception as e:
            caption = f"ERROR: {e}"
            n_err  += 1

            if "CUDA" in str(e).upper():
                results.append({
                    "image_path":       img_path,
                    "disease_category": disease,
                    "bbox_source":      bbox_source,
                    "generated_prompt": caption,
                })
                save_results(results)
                return results, n_err, n_skip

        results.append({
            "image_path":       img_path,
            "disease_category": disease,
            "bbox_source":      bbox_source,
            "generated_prompt": caption,
        })

        if (i + 1) % cfg.save_interval == 0:
            save_results(results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    save_results(results)
    return results, n_err, n_skip


# ---------- main ----------

def main():
    set_seed()

    print("=" * 70)
    print("  STEP 2: GENERATE MEDICAL CAPTIONS — MedGemma 27B IT")
    print("  Sursa bbox: doctor (ground truth) | yolo (silver) | none")
    print("  LOW MEMORY MODE: 32GB RAM + 24GB VRAM")
    print("=" * 70)

    with open(cfg.master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Total imagini: {len(metadata)}")

    # statistici bbox_source
    src_counts = Counter(m.get("bbox_source", "none") for m in metadata)
    for src, cnt in sorted(src_counts.items()):
        print(f"    {src:10s}: {cnt}")

    mdl, proc = load_model()
    results, n_err, n_skip = process_all(mdl, proc, metadata)

    good        = [r for r in results if not r["generated_prompt"].startswith("ERROR")]
    word_counts = [len(r["generated_prompt"].split()) for r in good]
    per_disease = Counter(r["disease_category"] for r in results)
    per_source  = Counter(r.get("bbox_source", "none") for r in results)

    print(f"\n  Total: {len(results)} | Errors: {n_err} | Skipped: {n_skip}")
    print(f"\n  Per disease:")
    for d, c in sorted(per_disease.items()):
        print(f"    {d:12s}: {c}")
    print(f"\n  Per bbox_source:")
    for s, c in sorted(per_source.items()):
        print(f"    {s:12s}: {c}")
    if word_counts:
        avg = sum(word_counts) / len(word_counts)
        print(f"\n  Words per caption: avg={avg:.0f}, min={min(word_counts)}, max={max(word_counts)}")

    print(f"\n  Saved: {cfg.output_json}")
    print("=" * 70)

    del mdl, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()