import gc
import json
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...utils.seed import set_seed


class Config:
    model_path = "models/medgemma-1.5-4b-it"
    master_json = "data/oct5k/metadata/_master.json"
    prompts_output = "data/oct5k/medgemma_prompts.json"

    max_new_tokens = 512
    save_every = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def _save_prompts(results):
    out = Path(cfg.prompts_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_medgemma():
    print(f"\n  Model: {cfg.model_path}")
    print(f"  Device: {cfg.device} | Precision: bfloat16")

    processor = AutoProcessor.from_pretrained(cfg.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM: {vram:.1f}GB / {total:.1f}GB")

    print("  Încărcat!\n")
    return model, processor


def build_text_prompt(meta):
    disease = str(meta.get("disease_category", "UNKNOWN")).upper()

    lines = [
        f"You are an expert ophthalmic image analyzer. Provide a highly detailed, objective morphological description of this retinal OCT segmentation mask ({disease}).",
        "Focus STRICTLY on describing the visual features, layer geometries, thickness, structural deformations, and lesions.",
        "Do NOT compute severity. Do NOT make clinical diagnoses, deductions, or treatment suggestions. Just describe what is physically present based on the image and the following metadata:",
    ]

    # Extragerea grosimii
    if meta.get("has_boundaries") and meta.get("boundaries"):
        b = meta["boundaries"]
        trt = b.get("total_retinal_thickness", {})
        if trt:
            lines.append(
                f"- Total Retinal Thickness (TRT): mean {trt.get('mean_px', 0):.1f}px (range {trt.get('min_px', 0)} - {trt.get('max_px', 0)}px).")

        regions = b.get("regions", {})
        if regions:
            reg = [f"{name} ({data.get('mean_thickness_px', 0):.1f}px)" for name, data in regions.items()]
            lines.append("- Layer Thicknesses: " + ", ".join(reg) + ".")

        ndeform = int(b.get("num_deformations", 0))
        if ndeform > 0:
            zones = b.get("deformation_zones", [])
            unique_zones = list(set([d.get('zone', 'unknown') for d in zones]))
            unique_types = list(set([d.get('type', 'deformation') for d in zones]))
            max_dev = max([d.get('deviation_from_mean_px', 0) for d in zones]) if zones else 0

            lines.append(
                f"- Structural Deformations: {ndeform} abnormal points detected. Types present: {', '.join(unique_types)}.")
            lines.append(
                f"- Affected Zones: Primarily in the {', '.join(unique_zones)} region with a maximum deviation of {max_dev:+.1f}px.")

    if meta.get("has_bounding_boxes") and int(meta.get("num_lesions", 0)) > 0:
        classes = ", ".join(list(set(meta.get("lesion_classes", []))))
        lines.append(
            f"- Lesions: {meta.get('num_lesions', 0)} marked lesions ({classes}), covering {meta.get('total_lesion_area_percent', 0):.1f}% of the area.")

        for les in meta.get("lesions", [])[:3]:
            layer = les.get("layer_correlation", {}).get("affected_layer", "unknown layer")
            zone = les.get("retinal_zone", "unknown zone")
            size = les.get("size_px", [0, 0])
            lines.append(f"  * {les.get('class', 'Lesion')} at {zone} (Layer: {layer}), size: {size[0]}x{size[1]}px.")
    else:
        lines.append(
            "- Lesions: No focal lesions explicitly marked. Structure follows normal layering or generic deformations.")

    # Formatare forțată pentru rezultate VLM clare
    lines.append(
        "\nInstruction: Write a single, fluent, descriptive paragraph integrating these facts with what you observe in the mask. "
        "Do not use bullet points in your response. Start directly with the description of the retinal structure."
    )

    return "\n".join(lines)


@torch.no_grad()
def generate(model, processor, messages, keys):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_len = inputs["input_ids"].shape[1]
    model_inputs = {k: inputs[k].to(model.device) for k in keys if k in inputs}

    out = model.generate(
        **model_inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    text = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return text


def process_all(model, processor, metadata):
    existing = {}
    if cfg.resume and Path(cfg.prompts_output).exists():
        try:
            with open(cfg.prompts_output, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = {
                    x["image_path"]: x["generated_prompt"]
                    for x in data
                    if isinstance(x, dict) and "image_path" in x and "generated_prompt" in x
                }
        except Exception as e:
            print(f"  WARNING resume: {e}")
        print(f"  Resume: {len(existing)} existente")

    results, skipped, errors = [], 0, 0

    for i, meta in enumerate(tqdm(metadata, desc="Generare")):
        path = meta["image_path"]
        disease = meta["disease_category"]

        if path in existing:
            results.append({
                "image_path": path,
                "disease_category": disease,
                "generated_prompt": existing[path],
            })
            skipped += 1
            continue

        text_prompt = build_text_prompt(meta)
        mask_path = meta.get("mask_rgb_path")

        try:
            content = []

            # Încarcă masca doar dacă există
            if mask_path and os.path.exists(mask_path):
                with Image.open(mask_path) as mk:
                    content.append({"type": "image", "image": mk.convert("RGB").copy()})

            content.append({"type": "text", "text": text_prompt})

            if any(c["type"] == "image" for c in content):
                msgs = [{"role": "user", "content": content}]
                prompt = generate(model, processor, msgs, ["input_ids", "attention_mask", "pixel_values"])
            else:
                msgs = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
                prompt = generate(model, processor, msgs, ["input_ids", "attention_mask"])

        except Exception as e:
            prompt = f"ERROR: {e}"
            errors += 1
            if "CUDA" in str(e).upper():
                results.append({"image_path": path, "disease_category": disease, "generated_prompt": prompt})
                _save_prompts(results)
                return results, errors, skipped

        results.append({
            "image_path": path,
            "disease_category": disease,
            "generated_prompt": prompt,
        })

        # Salvare performantă o dată la 50 pași
        if (i + 1) % cfg.save_every == 0:
            _save_prompts(results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Salvare finală
    _save_prompts(results)
    return results, errors, skipped


def main():
    set_seed()
    print("=" * 70)
    print("  STEP 2: GENERATE MEDICAL DESCRIPTIVE CAPTIONS (MASK-ONLY)")
    print("=" * 70)

    with open(cfg.master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Imagini: {len(metadata)}")

    model, processor = load_medgemma()
    results, errors, skipped = process_all(model, processor, metadata)

    good = [r for r in results if not r["generated_prompt"].startswith("ERROR")]
    lengths = [len(r["generated_prompt"].split()) for r in good]
    diseases = Counter(r["disease_category"] for r in results)

    print(f"\n  Total: {len(results)} | Erori: {errors} | Skip: {skipped}")
    for d, c in sorted(diseases.items()):
        print(f"    {d:12s}: {c}")
    if lengths:
        print(f"  Cuvinte per caption: medie={sum(lengths) / len(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")

    print(f"\n  Salvat: {cfg.prompts_output}")
    print("=" * 70)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()