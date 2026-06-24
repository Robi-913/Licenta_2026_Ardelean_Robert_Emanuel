"""
Step 3: Split MedGemma prompts in prompt_a (structura) si prompt_b (patologie)

Gemini 3.1 Flash-Lite — fara thinking, cost minim
  Input:  $0.25 / 1M tokens
  Output: $1.50 / 1M tokens

Rulare:
    $env:GEMINI_API_KEY="cheia_ta"
    python -m src.pipelines.medgemma.split_prompts_gemini
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from src.utils.seed import set_seed


# ---------- config ----------

class Config:
    api_key     = "AQ.Ab8RN6IwlD-57rBm1qVihCp_agLMoBlCujyXJOpR_J4tVAXs9g"
    model       = "gemini-3.1-flash-lite"

    input_json  = "data/OCT5k/medgemma_prompts_v2_27b.json"
    output_json = "data/oct5k/medgemma_prompts_split_v2_27b.json"

    save_interval  = 100
    resume         = True
    max_retries    = 2
    sleep_on_error = 2.0

    # pricing per 1M tokens
    price_input  = 0.25
    price_output = 1.50


cfg = Config()

DISEASE_FULL = {
    "AMD":    "Age-Related Macular Degeneration",
    "DME":    "Diabetic Macular Edema",
    "DRUSEN": "Drusen",
    "NORMAL": "Normal healthy retina",
}


# ---------- prompt ----------

SPLIT_PROMPT = """Split this OCT description into exactly two parts. Each part MUST:
- Start with "{disease}"
- Be a single coherent sentence of maximum 50 words
- Do NOT end with a period
- Pack as much clinical information as possible
- Remove all color words (blue, green, yellow, cyan) and "mask", "segmentation"
- Write naturally within the word limit, do NOT truncate mid-sentence

Part A - STRUCTURE: retinal layer thicknesses, boundaries, morphology, measurements, deformations. Start with "{disease}" then describe structure in one complete sentence without period at end.

Part B - PATHOLOGY: lesions, biomarkers, disease indicators, clinical findings. Start with "{disease}" then describe pathology in one complete sentence without period at end.

Input text:
{text}

Output EXACTLY 2 lines, nothing else:
A: {disease} ...structure...
B: {disease} ...pathology..."""


# ---------- helpers ----------

def save_results(data):
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_split(response_text, disease_full=""):
    lines = response_text.strip().split("\n")
    prompt_a, prompt_b = "", ""

    for line in lines:
        line = line.strip()
        if line.startswith("A:"):
            prompt_a = line[2:].strip()
        elif line.startswith("B:"):
            prompt_b = line[2:].strip()

    # asigura disease name la inceput
    if prompt_a and not prompt_a.startswith(disease_full):
        prompt_a = f"{disease_full} {prompt_a}"
    if prompt_b and not prompt_b.startswith(disease_full):
        prompt_b = f"{disease_full} {prompt_b}"

    # scoate punct final
    if prompt_a.endswith("."):
        prompt_a = prompt_a[:-1]
    if prompt_b.endswith("."):
        prompt_b = prompt_b[:-1]

    return prompt_a, prompt_b


# ---------- cost tracker ----------

class CostTracker:
    def __init__(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.n_calls = 0

    def add(self, usage_metadata):
        if usage_metadata:
            self.total_input_tokens  += getattr(usage_metadata, "prompt_token_count", 0)
            self.total_output_tokens += getattr(usage_metadata, "candidates_token_count", 0)
        self.n_calls += 1

    @property
    def cost_input(self):
        return self.total_input_tokens / 1_000_000 * cfg.price_input

    @property
    def cost_output(self):
        return self.total_output_tokens / 1_000_000 * cfg.price_output

    @property
    def cost_total(self):
        return self.cost_input + self.cost_output

    def summary(self):
        return (
            f"  API calls:      {self.n_calls}\n"
            f"  Input tokens:   {self.total_input_tokens:,} (${self.cost_input:.4f})\n"
            f"  Output tokens:  {self.total_output_tokens:,} (${self.cost_output:.4f})\n"
            f"  TOTAL COST:     ${self.cost_total:.4f}"
        )


# ---------- main ----------

def main():
    set_seed()

    print("=" * 70)
    print("  STEP 3: SPLIT PROMPTS (Structure / Pathology)")
    print(f"  Model: {cfg.model} (no thinking, cost-optimized)")
    print(f"  Input: {cfg.input_json}")
    print(f"  Pricing: ${cfg.price_input}/M input, ${cfg.price_output}/M output")
    print("=" * 70)

    if not cfg.api_key:
        cfg.api_key = input("  Gemini API key: ").strip()

    client = genai.Client(api_key=cfg.api_key)
    cost   = CostTracker()

    with open(cfg.input_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"  Total prompts: {len(prompts)}")

    # resume
    done = {}
    if cfg.resume and Path(cfg.output_json).exists():
        try:
            with open(cfg.output_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                done = prev
            elif isinstance(prev, list):
                done = {x["image_path"]: x for x in prev if isinstance(x, dict) and "image_path" in x}
        except Exception as e:
            print(f"  WARNING resume: {e}")
        print(f"  Resume: {len(done)} already done")

    results = dict(done)
    n_new = 0
    n_err = 0

    for i, entry in enumerate(tqdm(prompts, desc="  Splitting")):
        img_path = entry["image_path"]
        text     = entry.get("generated_prompt", "")
        disease  = entry.get("disease_category", "UNKNOWN").upper()
        disease_full = DISEASE_FULL.get(disease, disease)

        if img_path in done:
            continue

        if not text or text.startswith("ERROR"):
            results[img_path] = {
                "image_path": img_path,
                "a": f"{disease_full} retinal scan with no available description",
                "b": f"{disease_full} retinal scan with no pathological findings described",
            }
            continue

        prompt_a, prompt_b = "", ""

        for attempt in range(cfg.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=cfg.model,
                    contents=SPLIT_PROMPT.format(
                        text=text[:1500],
                        disease=disease_full,
                    ),
                    config=types.GenerateContentConfig(
                        max_output_tokens=512,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                        ),
                    ),
                )

                # track cost
                cost.add(getattr(response, "usage_metadata", None))

                # extrage text
                resp_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if part.text:
                        resp_text += part.text

                prompt_a, prompt_b = parse_split(resp_text, disease_full)

                if prompt_a and prompt_b:
                    break

            except Exception as e:
                if attempt < cfg.max_retries:
                    time.sleep(cfg.sleep_on_error)
                else:
                    prompt_a = f"{disease_full} " + " ".join(text.split()[:45])
                    prompt_b = f"{disease_full} " + " ".join(text.split()[45:90])
                    n_err += 1

        results[img_path] = {
            "image_path": img_path,
            "a": prompt_a,
            "b": prompt_b,
        }
        n_new += 1

        # print cost periodic
        if n_new % cfg.save_interval == 0:
            save_results(results)
            tqdm.write(f"  [{n_new}] Cost so far: ${cost.cost_total:.4f} "
                       f"({cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out)")

    save_results(results)

    # statistici finale
    n_good  = sum(1 for v in results.values() if v.get("a") and v.get("b"))
    n_empty = sum(1 for v in results.values() if not v.get("a") or not v.get("b"))

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"  Total: {len(results)} | New: {n_new} | Errors: {n_err}")
    print(f"  Good splits: {n_good} | Empty/partial: {n_empty}")
    print(f"\n  COST BREAKDOWN:")
    print(cost.summary())
    print(f"\n  Saved: {cfg.output_json}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()