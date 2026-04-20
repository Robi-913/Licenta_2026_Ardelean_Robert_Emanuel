import gc
import json
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...utils.seed import set_seed


class Config:
    model_path = "models/qwen2.5-7b-instruct"
    input_json = "data/oct5k/medgemma_prompts.json"
    output_json = "data/oct5k/medgemma_prompts_split.json"

    max_new_tokens = 300
    max_words_a = 50
    max_words_b = 50
    max_retry = 2

    save_every = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def load_medgemma():
    print(f"\n  Model: {cfg.model_path} (Text-Only Mode)")
    processor = AutoProcessor.from_pretrained(cfg.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor


SYSTEM_PROMPT = (
    "You are a strict text editor. Rewrite the following medical text into EXACTLY two short sentences for a raw grayscale OCT.\n"
    "RULES:\n"
    "1. Completely REMOVE the words 'mask', 'segmentation', and all colors (e.g., blue, yellow, green, cyan, light, dark).\n"
    "2. Do NOT add medical diagnoses or treatment advice.\n"
    "3. You MUST format your output exactly like this example:\n"
    "PROMPT_A: The overall retinal thickness is 85 pixels with the outer nerve fiber layer measuring 44 pixels.\n"
    "PROMPT_B: There are several structural deformities in the temporal region with four distinct lesions present."
)


def build_split_request(long_prompt):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Source Text:\n{long_prompt}\n\n"
        f"Instructions:\n"
        f"- PROMPT_A: Summarize ONLY the layer structure and thicknesses (max 45 words).\n"
        f"- PROMPT_B: Summarize ONLY the anomalies, lesions, and deformations (max 45 words).\n\n"
        f"Output exactly in this format:\nPROMPT_A: ...\nPROMPT_B: ..."
    )


@torch.no_grad()
def _generate_once(model, processor, long_prompt, retry_hint=""):
    text_prompt = build_split_request(long_prompt) + retry_hint

    # Format pur textual pentru MedGemma
    messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    model_inputs = {k: inputs[k].to(model.device) for k in ["input_ids", "attention_mask"] if k in inputs}
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **model_inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def _clean_line(s: str) -> str:
    s = " ".join((s or "").split())
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def parse_split(response, long_prompt):
    prompt_a, prompt_b = "", ""

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("PROMPT_A:"):
            prompt_a = line[len("PROMPT_A:"):].strip()
        elif line.upper().startswith("PROMPT_B:"):
            prompt_b = line[len("PROMPT_B:"):].strip()

    if not prompt_a or not prompt_b:
        words = long_prompt.split()
        mid = max(1, len(words) // 2)
        if not prompt_a: prompt_a = " ".join(words[:mid])
        if not prompt_b: prompt_b = " ".join(words[mid:])

    prompt_a = _clean_line(" ".join(prompt_a.split()[:cfg.max_words_a]))
    prompt_b = _clean_line(" ".join(prompt_b.split()[:cfg.max_words_b]))

    return prompt_a, prompt_b


def validate_split(prompt_a, prompt_b):
    issues = []
    if not prompt_a: issues.append("empty_a")
    if not prompt_b: issues.append("empty_b")
    if len(prompt_a.split()) > cfg.max_words_a: issues.append("a_too_long")
    if len(prompt_b.split()) > cfg.max_words_b: issues.append("b_too_long")

    banned_medical = ["treatment", "operation", "surgery", "prognosis", "intervention", "severity"]
    if any(w in prompt_a.lower() for w in banned_medical): issues.append("a_has_medical_opinion")
    if any(w in prompt_b.lower() for w in banned_medical): issues.append("b_has_medical_opinion")

    banned_visuals = ["mask", "segmentation", "color", "colored", "blue", "green", "yellow", "red", "cyan", "turquoise",
                      "navy", "cream", "hue", "band", "light", "dark"]
    a_words = set(re.findall(r'\b\w+\b', prompt_a.lower()))
    b_words = set(re.findall(r'\b\w+\b', prompt_b.lower()))

    if any(w in a_words for w in banned_visuals): issues.append("a_has_color_or_mask_ref")
    if any(w in b_words for w in banned_visuals): issues.append("b_has_color_or_mask_ref")

    return len(issues) == 0, issues


def split_prompt(model, processor, long_prompt):
    best = None
    best_issues = None

    for t in range(cfg.max_retry + 1):
        hint = "\n\nRetry: Follow the PROMPT_A and PROMPT_B format strictly and remove all color/mask words." if t > 0 else ""
        response = _generate_once(model, processor, long_prompt, retry_hint=hint)
        a, b = parse_split(response, long_prompt)
        valid, issues = validate_split(a, b)

        best = (a, b, response)
        best_issues = issues
        if valid:
            return a, b, True, []

    return best[0], best[1], False, best_issues


def _save(results):
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_all(model, processor, prompts_data):
    existing = {}
    if cfg.resume and Path(cfg.output_json).exists():
        with open(cfg.output_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        existing = {x["image_path"]: x for x in old if "image_path" in x}

    results, skipped, errors = [], 0, 0

    for i, item in enumerate(tqdm(prompts_data, desc="Split prompturi")):
        path = item["image_path"]
        disease = item["disease_category"]
        long_prompt = item["generated_prompt"]

        if long_prompt.startswith("ERROR"): continue
        if path in existing and existing[path].get("split_valid") == True:
            results.append(existing[path])
            skipped += 1
            continue

        try:
            prompt_a, prompt_b, is_valid, issues = split_prompt(model, processor, long_prompt)
        except Exception as e:
            errors += 1
            prompt_a, prompt_b, is_valid, issues = "", "", False, [f"exception:{str(e)}"]

        results.append({
            "image_path": path,
            "disease_category": disease,
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "original_prompt": long_prompt,
            "split_valid": is_valid,
            "split_issues": issues,
        })

        if (i + 1) % cfg.save_every == 0:
            _save(results)

    _save(results)
    return results, errors, skipped


def main():
    set_seed()
    with open(cfg.input_json, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    valid_data = [p for p in prompts_data if not p["generated_prompt"].startswith("ERROR")]
    model, processor = load_medgemma()
    results, errors, skipped = process_all(model, processor, valid_data)


if __name__ == "__main__":
    main()