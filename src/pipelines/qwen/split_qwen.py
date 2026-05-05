import gc
import json
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...utils.seed import set_seed


# ---------- config ----------

class Config:
    model_path = "models/qwen2.5-7b-instruct"
    src_json = "data/oct5k/medgemma_prompts.json"
    out_json = "data/oct5k/medgemma_prompts_split.json"

    max_tokens = 300
    limit_a = 50
    limit_b = 50
    retries = 2

    save_interval = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ---------- model ----------

def load_model():
    print(f"\n  Model: {cfg.model_path} (Text-Only Mode)")
    proc = AutoProcessor.from_pretrained(cfg.model_path)
    mdl = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    mdl.eval()
    return mdl, proc


# ---------- prompt construction ----------

SYS_PROMPT = (
    "You are a strict text editor. Rewrite the following medical text into EXACTLY two short sentences for a raw grayscale OCT.\n"
    "RULES:\n"
    "1. Completely REMOVE the words 'mask', 'segmentation', and all colors (e.g., blue, yellow, green, cyan, light, dark).\n"
    "2. Do NOT add medical diagnoses or treatment advice.\n"
    "3. You MUST format your output exactly like this example:\n"
    "PROMPT_A: The overall retinal thickness is 85 pixels with the outer nerve fiber layer measuring 44 pixels.\n"
    "PROMPT_B: There are several structural deformities in the temporal region with four distinct lesions present."
)


def make_request(long_text):
    return (
        f"{SYS_PROMPT}\n\n"
        f"Source Text:\n{long_text}\n\n"
        f"Instructions:\n"
        f"- PROMPT_A: Summarize ONLY the layer structure and thicknesses (max 45 words).\n"
        f"- PROMPT_B: Summarize ONLY the anomalies, lesions, and deformations (max 45 words).\n\n"
        f"Output exactly in this format:\nPROMPT_A: ...\nPROMPT_B: ..."
    )


# ---------- generation ----------

@torch.no_grad()
def call_model(mdl, proc, long_text, extra=""):
    full = make_request(long_text) + extra

    msgs = [{"role": "user", "content": [{"type": "text", "text": full}]}]

    inputs = proc.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    prefix = inputs["input_ids"].shape[1]
    feed = {k: inputs[k].to(mdl.device) for k in ["input_ids", "attention_mask"] if k in inputs}

    out = mdl.generate(
        **feed,
        max_new_tokens=cfg.max_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=proc.tokenizer.eos_token_id,
    )

    return proc.decode(out[0][prefix:], skip_special_tokens=True).strip()


# ---------- parsing / validation ----------

def clean(s):
    s = " ".join((s or "").split())
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def parse_response(response, fallback):
    pa, pb = "", ""

    for line in response.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("PROMPT_A:"):
            pa = line[len("PROMPT_A:"):].strip()
        elif upper.startswith("PROMPT_B:"):
            pb = line[len("PROMPT_B:"):].strip()

    if not pa or not pb:
        words = fallback.split()
        mid = max(1, len(words) // 2)
        if not pa:
            pa = " ".join(words[:mid])
        if not pb:
            pb = " ".join(words[mid:])

    pa = clean(" ".join(pa.split()[:cfg.limit_a]))
    pb = clean(" ".join(pb.split()[:cfg.limit_b]))

    return pa, pb


BANNED_MED = ["treatment", "operation", "surgery", "prognosis", "intervention", "severity"]
BANNED_VIS = [
    "mask", "segmentation", "color", "colored",
    "blue", "green", "yellow", "red", "cyan", "turquoise",
    "navy", "cream", "hue", "band", "light", "dark",
]


def check(pa, pb):
    problems = []

    if not pa:
        problems.append("empty_a")
    if not pb:
        problems.append("empty_b")
    if len(pa.split()) > cfg.limit_a:
        problems.append("a_too_long")
    if len(pb.split()) > cfg.limit_b:
        problems.append("b_too_long")

    if any(w in pa.lower() for w in BANNED_MED):
        problems.append("a_has_medical_opinion")
    if any(w in pb.lower() for w in BANNED_MED):
        problems.append("b_has_medical_opinion")

    wa = set(re.findall(r"\b\w+\b", pa.lower()))
    wb = set(re.findall(r"\b\w+\b", pb.lower()))

    if wa & set(BANNED_VIS):
        problems.append("a_has_color_or_mask_ref")
    if wb & set(BANNED_VIS):
        problems.append("b_has_color_or_mask_ref")

    return len(problems) == 0, problems


def do_split(mdl, proc, long_text):
    best_a, best_b = "", ""
    best_problems = None

    for attempt in range(cfg.retries + 1):
        hint = ""
        if attempt > 0:
            hint = "\n\nRetry: Follow the PROMPT_A and PROMPT_B format strictly and remove all color/mask words."

        raw = call_model(mdl, proc, long_text, extra=hint)
        pa, pb = parse_response(raw, long_text)
        ok, problems = check(pa, pb)

        best_a, best_b = pa, pb
        best_problems = problems

        if ok:
            return pa, pb, True, []

    return best_a, best_b, False, best_problems


# ---------- batch processing ----------

def save_out(data):
    out = Path(cfg.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_all(mdl, proc, data):
    prev = {}
    if cfg.resume and Path(cfg.out_json).exists():
        with open(cfg.out_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        prev = {x["image_path"]: x for x in old if "image_path" in x}

    results = []
    n_skip = 0
    n_err = 0

    for i, item in enumerate(tqdm(data, desc="Splitting")):
        path = item["image_path"]
        disease = item["disease_category"]
        long_text = item["generated_prompt"]

        if long_text.startswith("ERROR"):
            continue

        cached = prev.get(path)
        if cached and cached.get("split_valid") is True:
            results.append(cached)
            n_skip += 1
            continue

        try:
            pa, pb, ok, problems = do_split(mdl, proc, long_text)
        except Exception as e:
            n_err += 1
            pa, pb, ok, problems = "", "", False, [f"exception:{e}"]

        results.append({
            "image_path": path,
            "disease_category": disease,
            "prompt_a": pa,
            "prompt_b": pb,
            "original_prompt": long_text,
            "split_valid": ok,
            "split_issues": problems,
        })

        if (i + 1) % cfg.save_interval == 0:
            save_out(results)

    save_out(results)
    return results, n_err, n_skip


# ---------- main ----------

def main():
    set_seed()

    with open(cfg.src_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    usable = [p for p in raw if not p["generated_prompt"].startswith("ERROR")]

    mdl, proc = load_model()
    results, n_err, n_skip = run_all(mdl, proc, usable)


if __name__ == "__main__":
    main()