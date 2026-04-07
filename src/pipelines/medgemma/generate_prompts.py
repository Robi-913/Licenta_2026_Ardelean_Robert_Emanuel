import gc
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ...utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # model local
    model_path = "models/medgemma-4b-it"

    # input metadata (din Step 1)
    master_json = "data/oct5k/metadata/_master.json"

    # output
    prompts_output = "data/oct5k/medgemma_prompts.json"
    prompts_dir = "data/oct5k/prompts"

    # generation
    max_new_tokens = 300
    temperature = 0.1
    top_p = 0.9
    do_sample = False  # <- schimbat conform cerinței

    # alege EXACT una:
    use_fp16 = True
    use_8bit = False
    use_4bit = False

    save_every = 50
    resume = True

    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════

def _validate_precision_config():
    flags = [cfg.use_fp16, cfg.use_8bit, cfg.use_4bit]
    if sum(bool(x) for x in flags) != 1:
        raise ValueError("Config invalid: setează exact una dintre use_fp16/use_8bit/use_4bit.")


def _precision_mode_str() -> str:
    if cfg.use_4bit:
        return "4-bit"
    if cfg.use_8bit:
        return "8-bit"
    return "bf16 (cfg.use_fp16=True)"


def _get_model_cuda_device_index(model) -> int:
    if not torch.cuda.is_available():
        return -1
    try:
        p = next(model.parameters())
        if p.is_cuda:
            return p.device.index if p.device.index is not None else 0
    except Exception:
        pass
    return 0


def _get_total_vram_gb(device_index: int) -> float:
    if device_index is None or device_index < 0:
        return 0.0
    props = torch.cuda.get_device_properties(device_index)
    total_bytes = getattr(props, "total_memory", None)
    if total_bytes is None:
        return 0.0
    return float(total_bytes) / (1024 ** 3)


def _safe_cuda_empty_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _save_prompts(results):
    out_path = Path(cfg.prompts_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _save_prompt_backup(image_path: str, generated_prompt: str, disease_category: str):
    # backup: un JSON per imagine
    # nume fișier safe din image_path
    safe_name = image_path.replace("\\", "_").replace("/", "_").replace(":", "_")
    out_file = Path(cfg.prompts_dir) / f"{safe_name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": image_path,
        "disease_category": disease_category,
        "generated_prompt": generated_prompt,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_medgemma():
    _validate_precision_config()

    print(f"\n  Încărcare MedGemma din: {cfg.model_path}")
    print(f"  Device: {cfg.device}")
    print(f"  Precision: {_precision_mode_str()}")

    quant_config = None
    if cfg.use_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # mai stabil decât float16 în multe cazuri
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    model_kwargs = {"device_map": "auto"}

    # Cerință explicită: în loc de torch_dtype=float16 -> dtype=bfloat16
    if cfg.use_fp16:
        model_kwargs["dtype"] = torch.bfloat16

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForImageTextToText.from_pretrained(cfg.model_path, **model_kwargs)
    model.eval()

    if torch.cuda.is_available():
        try:
            idx = _get_model_cuda_device_index(model)
            used = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            total = _get_total_vram_gb(idx)
            name = torch.cuda.get_device_name(idx)
            print(f"  GPU[{idx}] {name} | VRAM: {used:.1f}GB / {total:.1f}GB")
        except Exception as e:
            print(f"  (VRAM report skipped: {e})")

    print("  Model încărcat!\n")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════

def build_text_prompt(meta: dict) -> str:
    disease = meta["disease_category"]
    lines = [
        f"This is a retinal layer segmentation mask from an OCT B-scan of a patient diagnosed with {disease}."
    ]

    if meta.get("has_boundaries") and meta.get("boundaries"):
        b = meta["boundaries"]
        trt = b["total_retinal_thickness"]
        lines.append(
            f"Total retinal thickness: mean {trt['mean_px']:.0f}px "
            f"({trt['mean_pct']:.1f}% of scan height), "
            f"range {trt['min_px']}-{trt['max_px']}px, std {trt['std_px']:.1f}px."
        )

        regions = b.get("regions", {})
        if regions:
            reg = [
                f"{name}: {data['mean_thickness_px']:.0f}px (std={data['std_thickness_px']:.1f})"
                for name, data in regions.items()
            ]
            lines.append("Layer thicknesses: " + ", ".join(reg) + ".")

        if b.get("num_deformations", 0) > 0:
            lines.append(f"Detected {b['num_deformations']} zones with abnormal thickness deviation.")
            for d in b.get("deformation_zones", [])[:5]:
                lines.append(
                    f"  - {d['type']} at x={d['x_normalized']:.2f} ({d['zone']}), "
                    f"thickness={d['thickness_px']}px, deviation={d['deviation_from_mean_px']:+.0f}px from mean."
                )

    if meta.get("has_bounding_boxes") and meta.get("num_lesions", 0) > 0:
        lines.append(
            f"Detected {meta['num_lesions']} lesion(s): {', '.join(meta.get('lesion_classes', []))}. "
            f"Total lesion area: {meta.get('total_lesion_area_percent', 0.0):.1f}% of image."
        )
        for i, les in enumerate(meta.get("lesions", [])[:8]):
            layer = les.get("layer_correlation", {})
            lines.append(
                f"  Lesion {i + 1}: {les['class']} at "
                f"x={les['center_normalized'][0]:.2f}, y={les['center_normalized'][1]:.2f} "
                f"({les['retinal_zone']}), size {les['size_px'][0]}x{les['size_px'][1]}px "
                f"({les['area_percent']:.2f}% area). Depth: {layer.get('depth_info', 'unknown')}."
            )
    else:
        lines.append(
            "No lesions detected. The retinal layers appear intact."
            if disease == "NORMAL"
            else "No bounding box annotations available for this image."
        )

    lines += [
        "",
        "Based on the segmentation mask shown and the structural information above, "
        "generate a detailed clinical description of this OCT scan. "
        "Describe the condition of each retinal layer, any pathological findings, "
        "the severity and location of abnormalities, and their clinical significance. "
        "Be specific about which layers are affected and how.",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_from_messages(model, processor, messages, input_keys):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    input_len = inputs["input_ids"].shape[1]
    model_inputs = {k: inputs[k].to(model.device) for k in input_keys if k in inputs}

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
    )

    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_prompt(model, processor, mask_image, text_prompt):
    mask_image = mask_image.resize((448, 448), Image.LANCZOS)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": mask_image},
            {"type": "text", "text": text_prompt},
        ],
    }]
    return _generate_from_messages(
        model, processor, messages, input_keys=["input_ids", "attention_mask", "pixel_values"]
    )


@torch.no_grad()
def generate_prompt_text_only(model, processor, text_prompt):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": text_prompt}],
    }]
    return _generate_from_messages(
        model, processor, messages, input_keys=["input_ids", "attention_mask"]
    )


# ═══════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def process_all_images(model, processor, all_metadata):
    Path(cfg.prompts_dir).mkdir(parents=True, exist_ok=True)

    existing_prompts = {}
    if cfg.resume and Path(cfg.prompts_output).exists():
        with open(cfg.prompts_output, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_prompts = {x["image_path"]: x["generated_prompt"] for x in existing_data}
        print(f"  Resume: {len(existing_prompts)} prompturi existente")

    results, skipped, errors = [], 0, 0
    start_time = time.time()

    pbar = tqdm(all_metadata, desc="Generare prompturi")
    for i, meta in enumerate(pbar):
        img_path = meta["image_path"]
        disease = meta["disease_category"]

        if img_path in existing_prompts:
            generated = existing_prompts[img_path]
            results.append({"image_path": img_path, "disease_category": disease, "generated_prompt": generated})
            skipped += 1
            continue

        text_prompt = build_text_prompt(meta)
        mask_path = meta.get("mask_rgb_path")
        generated = ""

        try:
            if mask_path and os.path.exists(mask_path):
                with Image.open(mask_path) as im:
                    mask_img = im.convert("RGB")
                generated = generate_prompt(model, processor, mask_img, text_prompt)
            else:
                generated = generate_prompt_text_only(model, processor, text_prompt)

        except Exception as e:
            errors += 1
            generated = f"ERROR: {e}"
            print(f"\n  Eroare la {img_path}: {e}")

            if "CUDA" in str(e).upper():
                print("  CUDA error detectat. Salvăm progresul și oprim.")
                results.append({"image_path": img_path, "disease_category": disease, "generated_prompt": generated})
                _save_prompts(results)
                _save_prompt_backup(img_path, generated, disease)
                return results, errors, skipped

        results.append({"image_path": img_path, "disease_category": disease, "generated_prompt": generated})
        _save_prompt_backup(img_path, generated, disease)

        elapsed = time.time() - start_time
        processed_now = (i + 1 - skipped)
        rate = processed_now / max(1e-6, elapsed)
        remaining = (len(all_metadata) - i - 1) / max(1e-6, rate)

        pbar.set_postfix(
            errors=errors,
            rate=f"{rate:.2f} img/s",
            eta=f"{remaining/60:.1f} min",
        )

        if (i + 1) % cfg.save_every == 0:
            _save_prompts(results)
            _safe_cuda_empty_cache()

    _save_prompts(results)
    return results, errors, skipped


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(results, errors, skipped):
    print(f"\n  {'─' * 55}")
    print(f"  Total: {len(results)}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Erori: {errors}")

    disease_counts = Counter(r["disease_category"] for r in results)
    print("\n  Per boală:")
    for d, c in sorted(disease_counts.items()):
        print(f"    {d:12s}: {c}")

    lengths = [
        len(r["generated_prompt"].split())
        for r in results
        if not r["generated_prompt"].startswith("ERROR")
    ]
    if lengths:
        print("\n  Lungime prompturi (cuvinte):")
        print(f"    Medie: {sum(lengths)/len(lengths):.0f}")
        print(f"    Min:   {min(lengths)}")
        print(f"    Max:   {max(lengths)}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    print("=" * 70)
    print("  STEP 2: GENERATE MEDICAL PROMPTS WITH MEDGEMMA")
    print("=" * 70)

    print(f"\n  Citire metadata: {cfg.master_json}")
    with open(cfg.master_json, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    print(f"  Total imagini: {len(all_metadata)}")

    model, processor = load_medgemma()
    results, errors, skipped = process_all_images(model, processor, all_metadata)
    print_report(results, errors, skipped)

    print(f"\n{'=' * 70}")
    print("  STEP 2 COMPLETE!")
    print(f"  Prompturi salvate: {cfg.prompts_output}")
    print(f"  Backup per imagine: {cfg.prompts_dir}")
    print(f"  Total: {len(results)} | Erori: {errors}")
    print(f"{'=' * 70}")

    del model, processor
    gc.collect()
    _safe_cuda_empty_cache()


if __name__ == "__main__":
    main()