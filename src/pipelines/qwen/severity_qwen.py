import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURARE
# ═══════════════════════════════════════════════════════════════════════

class Config:
    model_path = "models/qwen2.5-7b-instruct"
    input_json = "data/oct5k/medgemma_prompts.json"      # prompturile originale complete
    output_json = "data/oct5k/severity_scores.json"

    max_new_tokens = 250
    max_retry = 2

    save_every = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pt NORMAL: severity random intre 0-10%
    normal_severity_min = 0
    normal_severity_max = 10


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════
# INCARCARE MODEL
# ═══════════════════════════════════════════════════════════════════════

def load_qwen():
    print(f"\n  Model: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - SEVERITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a retinal disease severity estimator for OCT scans.\n"
    "Analyze the description and estimate how severe the eye condition is.\n\n"
    "Consider these factors:\n"
    "- Number of lesions (more = worse)\n"
    "- Type of lesions (PED, SRF, IRF are more severe than simple drusen)\n"
    "- Size of lesions (larger = worse)\n"
    "- Number of abnormal/deformation points (more = worse)\n"
    "- Magnitude of thickness deviation (larger deviation = worse)\n"
    "- Area coverage percentage (higher = worse)\n\n"
    "Severity scale:\n"
    "- 0-15%: Minimal findings, healthy retina with normal variations\n"
    "- 15-30%: Mild, small drusen or minor deformations, not alarming\n"
    "- 30-50%: Moderate, multiple drusen or notable structural changes\n"
    "- 50-70%: Significant, large/numerous lesions, fluid present\n"
    "- 70-85%: Severe, extensive damage, multiple pathological features\n"
    "- 85-100%: Critical, massive structural disruption, urgent\n\n"
    "You MUST output EXACTLY in this format (3 lines, nothing else):\n"
    "Reasoning: <brief explanation of key findings, max 2 sentences>\n"
    "Level: <one of: Minimal, Mild, Moderate, Significant, Severe, Critical>\n"
    "Severity: <number>%"
)


# ═══════════════════════════════════════════════════════════════════════
# GENERARE SI PARSARE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_once(model, tokenizer, original_prompt, retry_hint=""):
    user_msg = (
        f"Analyze the severity of this OCT scan description:\n\n"
        f"{original_prompt}\n\n"
        f"Output exactly:\n"
        f"Reasoning: ...\n"
        f"Level: ...\n"
        f"Severity: ...%"
        f"{retry_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def parse_severity(response):
    """Extrage severity_percent, level si reasoning din raspunsul modelului."""
    severity_percent = None
    level = None
    reasoning = None

    for line in response.split("\n"):
        line = line.strip()

        # Cauta "Severity: XX%"
        if line.lower().startswith("severity:"):
            match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
            if match:
                severity_percent = float(match.group(1))
                severity_percent = max(0, min(100, severity_percent))  # clamp 0-100

        # Cauta "Level: Moderate" etc
        elif line.lower().startswith("level:"):
            level = line.split(":", 1)[1].strip()

        # Cauta "Reasoning: ..."
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    return severity_percent, level, reasoning


def estimate_severity(model, tokenizer, original_prompt):
    """Incearca sa extraga severity cu retry logic."""
    best_response = None

    for t in range(cfg.max_retry + 1):
        hint = "\n\nRetry: Output EXACTLY 3 lines: Reasoning, Level, Severity." if t > 0 else ""
        response = _generate_once(model, tokenizer, original_prompt, retry_hint=hint)
        best_response = response

        sev_pct, level, reasoning = parse_severity(response)

        if sev_pct is not None:
            return sev_pct, level or "Unknown", reasoning or "", True, []

    # fallback: n-a reusit sa extraga un numar valid
    return None, None, best_response, False, ["parse_failed"]


# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════

def _save(results):
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

def process_all(model, tokenizer, prompts_data):
    # Resume logic
    existing = {}
    if cfg.resume and Path(cfg.output_json).exists():
        with open(cfg.output_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        existing = {x["image_path"]: x for x in old if x.get("severity_valid") == True}
        print(f"  Deja procesate valid: {len(existing)} imagini.")

    results = list(existing.values())
    processed, errors = 0, 0

    for i, item in enumerate(tqdm(prompts_data, desc="Severity scoring")):
        path = item["image_path"]
        disease = item["disease_category"]
        original = item["generated_prompt"]

        # Skip daca deja procesat
        if path in existing:
            continue

        # Skip daca promptul original e eroare
        if not original or original.startswith("ERROR"):
            continue

        # NORMAL => severity random 0-10%, fara LLM
        if disease.upper() == "NORMAL":
            sev = round(random.uniform(cfg.normal_severity_min, cfg.normal_severity_max), 1)
            results.append({
                "image_path": path,
                "disease_category": disease,
                "severity_percent": sev,
                "severity_level": "Minimal",
                "severity_reasoning": "Normal retina with physiological variations only.",
                "severity_valid": True,
                "severity_issues": [],
            })
            continue

        # Procesare cu Qwen pt AMD, DME, DRUSEN
        try:
            sev_pct, level, reasoning, is_valid, issues = estimate_severity(
                model, tokenizer, original
            )
        except Exception as e:
            errors += 1
            sev_pct, level, reasoning = None, None, str(e)
            is_valid, issues = False, [f"exception:{str(e)}"]

        results.append({
            "image_path": path,
            "disease_category": disease,
            "severity_percent": sev_pct,
            "severity_level": level,
            "severity_reasoning": reasoning,
            "severity_valid": is_valid,
            "severity_issues": issues,
        })
        processed += 1

        # Save periodic
        if processed % cfg.save_every == 0:
            _save(results)
            tqdm.write(f"  💾 Saved {len(results)} | Processed: {processed} | Errors: {errors}")

    _save(results)
    return results, errors


def main():
    random.seed(42)  # reproducibilitate pt NORMAL

    print("=" * 70)
    print("  STEP 3: SEVERITY SCORING CU QWEN2.5-7B-INSTRUCT (LOCAL)")
    print("  Cost: 0 RON (ruleaza pe GPU local)")
    print("=" * 70)

    # Incarca prompturile originale (generated_prompt = textul complet)
    with open(cfg.input_json, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    # Filtrare
    valid_data = [p for p in prompts_data if not p["generated_prompt"].startswith("ERROR")]
    print(f"  Total imagini: {len(prompts_data)}")
    print(f"  Valide: {len(valid_data)}")

    normal_count = sum(1 for p in valid_data if p["disease_category"].upper() == "NORMAL")
    disease_count = len(valid_data) - normal_count
    print(f"  NORMAL (hardcoded 0-10%): {normal_count}")
    print(f"  Disease (Qwen inference): {disease_count}")
    print(f"  Timp estimat: ~{disease_count // 60} min pe RTX 3090")

    # Incarca modelul
    model, tokenizer = load_qwen()

    # Proceseaza
    results, errors = process_all(model, tokenizer, valid_data)

    # Statistici finale
    valid_results = [r for r in results if r.get("severity_valid") == True]
    levels = [r["severity_level"] for r in valid_results if r.get("severity_level")]
    level_counts = dict(sorted(Counter(levels).items()))

    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS:")
    print(f"  Total procesate: {len(results)}")
    print(f"  Valide: {len(valid_results)}")
    print(f"  Erori: {errors}")
    print(f"  Distributie severity levels: {level_counts}")

    # Statistici per disease category
    for cat in ["NORMAL", "DRUSEN", "AMD", "DME"]:
        cat_results = [r for r in valid_results if r["disease_category"].upper() == cat]
        if cat_results:
            sevs = [r["severity_percent"] for r in cat_results if r["severity_percent"] is not None]
            if sevs:
                avg_sev = sum(sevs) / len(sevs)
                min_sev = min(sevs)
                max_sev = max(sevs)
                print(f"  {cat}: {len(cat_results)} img | avg: {avg_sev:.1f}% | min: {min_sev:.1f}% | max: {max_sev:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()