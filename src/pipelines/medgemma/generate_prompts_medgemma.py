import gc
import json
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.utils.seed import set_seed


# ---------- config ----------

class Config:
    model_path = "models/medgemma-1.5-4b-it"
    master_json = "data/oct5k/metadata/_master.json"
    output_json = "data/oct5k/medgemma_prompts.json"

    max_tokens = 512
    save_interval = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ---------- helpers ----------

def save_results(data):
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_model():
    print(f"\n  Model: {cfg.model_path}")
    print(f"  Device: {cfg.device} | Precision: bfloat16")

    proc = AutoProcessor.from_pretrained(cfg.model_path)
    mdl = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    mdl.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM: {used:.1f}GB / {total:.1f}GB")

    print("  Loaded!\n")
    return mdl, proc


# ---------- prompt builder ----------

def build_prompt(meta):
    disease = str(meta.get("disease_category", "UNKNOWN")).upper()

    parts = [
        f"You are an expert ophthalmic image analyzer. Provide a highly detailed, objective morphological description of this retinal OCT segmentation mask ({disease}).",
        "Focus STRICTLY on describing the visual features, layer geometries, thickness, structural deformations, and lesions.",
        "Do NOT compute severity. Do NOT make clinical diagnoses, deductions, or treatment suggestions. Just describe what is physically present based on the image and the following metadata:",
    ]

    has_bounds = meta.get("has_boundaries") and meta.get("boundaries")
    if has_bounds:
        b = meta["boundaries"]

        trt = b.get("total_retinal_thickness", {})
        if trt:
            parts.append(
                f"- Total Retinal Thickness (TRT): mean {trt.get('mean_px', 0):.1f}px "
                f"(range {trt.get('min_px', 0)} - {trt.get('max_px', 0)}px)."
            )

        regs = b.get("regions", {})
        if regs:
            items = [f"{name} ({d.get('mean_thickness_px', 0):.1f}px)" for name, d in regs.items()]
            parts.append("- Layer Thicknesses: " + ", ".join(items) + ".")

        n_def = int(b.get("num_deformations", 0))
        if n_def > 0:
            zones = b.get("deformation_zones", [])
            zone_set = list(set(d.get("zone", "unknown") for d in zones))
            type_set = list(set(d.get("type", "deformation") for d in zones))
            max_dev = max((d.get("deviation_from_mean_px", 0) for d in zones), default=0)

            parts.append(
                f"- Structural Deformations: {n_def} abnormal points detected. "
                f"Types present: {', '.join(type_set)}."
            )
            parts.append(
                f"- Affected Zones: Primarily in the {', '.join(zone_set)} region "
                f"with a maximum deviation of {max_dev:+.1f}px."
            )

    n_les = int(meta.get("num_lesions", 0))
    has_les = meta.get("has_bounding_boxes") and n_les > 0

    if has_les:
        cls_list = ", ".join(sorted(set(meta.get("lesion_classes", []))))
        area = meta.get("total_lesion_area_percent", 0)
        parts.append(f"- Lesions: {n_les} marked lesions ({cls_list}), covering {area:.1f}% of the area.")

        for les in meta.get("lesions", [])[:3]:
            layer = les.get("layer_correlation", {}).get("affected_layer", "unknown layer")
            zone = les.get("retinal_zone", "unknown zone")
            sz = les.get("size_px", [0, 0])
            parts.append(
                f"  * {les.get('class', 'Lesion')} at {zone} "
                f"(Layer: {layer}), size: {sz[0]}x{sz[1]}px."
            )
    else:
        parts.append(
            "- Lesions: No focal lesions explicitly marked. "
            "Structure follows normal layering or generic deformations."
        )

    parts.append(
        "\nInstruction: Write a single, fluent, descriptive paragraph integrating "
        "these facts with what you observe in the mask. "
        "Do not use bullet points in your response. "
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
    return decoded.strip()


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
                    if isinstance(x, dict) and "image_path" in x and "generated_prompt" in x
                }
        except Exception as e:
            print(f"  WARNING resume: {e}")
        print(f"  Resume: {len(done)} already done")

    results = []
    n_skip = 0
    n_err = 0

    for i, meta in enumerate(tqdm(metadata, desc="Generating")):
        img_path = meta["image_path"]
        disease = meta["disease_category"]

        if img_path in done:
            results.append({
                "image_path": img_path,
                "disease_category": disease,
                "generated_prompt": done[img_path],
            })
            n_skip += 1
            continue

        txt = build_prompt(meta)
        mask_file = meta.get("mask_rgb_path")

        try:
            content = []

            if mask_file and os.path.exists(mask_file):
                with Image.open(mask_file) as mk:
                    content.append({"type": "image", "image": mk.convert("RGB").copy()})

            content.append({"type": "text", "text": txt})

            has_img = any(c["type"] == "image" for c in content)
            keys = ["input_ids", "attention_mask"]
            if has_img:
                keys.append("pixel_values")

            msgs = [{"role": "user", "content": content}]
            caption = run_generate(mdl, proc, msgs, keys)

        except Exception as e:
            caption = f"ERROR: {e}"
            n_err += 1

            if "CUDA" in str(e).upper():
                results.append({
                    "image_path": img_path,
                    "disease_category": disease,
                    "generated_prompt": caption,
                })
                save_results(results)
                return results, n_err, n_skip

        results.append({
            "image_path": img_path,
            "disease_category": disease,
            "generated_prompt": caption,
        })

        if (i + 1) % cfg.save_interval == 0:
            save_results(results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_results(results)
    return results, n_err, n_skip


# ---------- main ----------

def main():
    set_seed()
    print("=" * 70)
    print("  STEP 2: GENERATE MEDICAL DESCRIPTIVE CAPTIONS (MASK-ONLY)")
    print("=" * 70)

    with open(cfg.master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Images: {len(metadata)}")

    mdl, proc = load_model()
    results, n_err, n_skip = process_all(mdl, proc, metadata)

    good = [r for r in results if not r["generated_prompt"].startswith("ERROR")]
    word_counts = [len(r["generated_prompt"].split()) for r in good]
    per_disease = Counter(r["disease_category"] for r in results)

    print(f"\n  Total: {len(results)} | Errors: {n_err} | Skipped: {n_skip}")
    for d, c in sorted(per_disease.items()):
        print(f"    {d:12s}: {c}")
    if word_counts:
        avg = sum(word_counts) / len(word_counts)
        print(f"  Words per caption: avg={avg:.0f}, min={min(word_counts)}, max={max(word_counts)}")

    print(f"\n  Saved: {cfg.output_json}")
    print("=" * 70)

    del mdl, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()