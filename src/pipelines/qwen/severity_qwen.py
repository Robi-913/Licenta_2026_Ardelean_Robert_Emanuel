"""
[ARCHIVED / TESTED - NOT MOVING FORWARD]

Acesta este un script vechi, pastrat ca dovada (proof of concept) ca s-a incercat
si varianta de a calcula scorul de severitate direct din text folosind Qwen2.5-7B-Instruct.
A fost inlocuit in pipeline-ul principal de abordarea determinista cu MedGemma (Severity Scoring v2).
NU este folosit mai departe in logica de business.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIGURARI

class Config:
    # Setarile pentru modelul Qwen folosit ca test local
    model_path = "model/qwen2.5-7b-instruct"
    src_json = "data/oct5k/medgemma_prompts.json"
    out_json = "data/oct5k/severity_scores.json"

    # Parametrii pentru generare
    max_tokens = 250
    retries = 2

    save_interval = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Marjele hardcodate pentru ochii sanatosi (fara boala)
    normal_min = 0
    normal_max = 10


cfg = Config()

# INCARCARE MODEL

def load_model():
    print(f"\n  Model: {cfg.model_path}")
    # Incarcam tokenizer-ul si modelul efectiv
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16,  # Folosim bfloat16 pentru a eficientiza consumul de memorie video
        device_map="auto",  # Lasa biblioteca sa imparta automat modelul pe GPU/CPU
    )
    mdl.eval()  # Oprim antrenarea (mod inferenta)

    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
    return mdl, tok


# PROMPT-UL SISTEMULUI (Regulile pentru Qwen)

SYS_PROMPT = (
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

# GENERARE SI PARSARE
@torch.no_grad()
def call_model(mdl, tok, description, extra=""):
    # Formateaza mesajul userului incluzand descrierea OCT si formatul strict
    user_msg = (
        f"Analyze the severity of this OCT scan description:\n\n"
        f"{description}\n\n"
        f"Output exactly:\n"
        f"Reasoning: ...\n"
        f"Level: ...\n"
        f"Severity: ...%"
        f"{extra}"
    )

    # Impachetam in template-ul oficial de chat (system msg + user msg)
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    prefix = inputs["input_ids"].shape[1]

    # Generam raspunsul modelului
    out = mdl.generate(
        **inputs,
        max_new_tokens=cfg.max_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
    )

    # Taiem promptul original din raspuns si il returnam curat
    return tok.decode(out[0][prefix:], skip_special_tokens=True).strip()


def parse(response):
    # Functie care cauta cuvintele cheie in raspunsul lui Qwen
    sev = None
    level = None
    reason = None

    for line in response.split("\n"):
        line = line.strip()
        low = line.lower()

        # Extragem severitatea (numarul in sine)
        if low.startswith("severity:"):
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if m:
                sev = max(0, min(100, float(m.group(1))))

        # Extragem categoria textuala (ex: Mild, Severe)
        elif low.startswith("level:"):
            level = line.split(":", 1)[1].strip()

        # Extragem rationamentul text
        elif low.startswith("reasoning:"):
            reason = line.split(":", 1)[1].strip()

    return sev, level, reason


def score_one(mdl, tok, description):
    last_raw = None

    # Mecanism de retry in caz ca modelul nu respecta formatul din prima
    for t in range(cfg.retries + 1):
        hint = ""
        if t > 0:
            hint = "\n\nRetry: Output EXACTLY 3 lines: Reasoning, Level, Severity."

        raw = call_model(mdl, tok, description, extra=hint)
        last_raw = raw

        # Incercam sa parsam. Daca avem severitatea, am reusit.
        sev, level, reason = parse(raw)
        if sev is not None:
            return sev, level or "Unknown", reason or "", True, []

    # Daca a esuat de toate datile, intoarcem eroare
    return None, None, last_raw, False, ["parse_failed"]

# FUNCTII DE SALVARE
def save_out(data):
    # Salveaza progresul pe disc sub forma de fisier JSON
    out = Path(cfg.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# BUCLA PRINCIPALA DE RULARE

def run_all(mdl, tok, data):
    # Logica de resume pt a nu re-procesa imagini deja calculate
    prev = {}
    if cfg.resume and Path(cfg.out_json).exists():
        with open(cfg.out_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        prev = {x["image_path"]: x for x in old if x.get("severity_valid") is True}
        print(f"  Already done: {len(prev)} images")

    results = list(prev.values())
    done = 0
    n_err = 0

    for i, item in enumerate(tqdm(data, desc="Severity scoring")):
        path = item["image_path"]
        disease = item["disease_category"]
        desc = item["generated_prompt"]

        if path in prev:
            continue

        if not desc or desc.startswith("ERROR"):
            continue

        # Daca poza e sanatoasa (NORMAL), nu mai deranjam reteaua neuronala;
        # hardcodam direct un scor foarte mic pt a economisi zeci de minute de GPU
        if disease.upper() == "NORMAL":
            sev = round(random.uniform(cfg.normal_min, cfg.normal_max), 1)
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

        # Pentru cele cu boli, cerem scorul de la Qwen
        try:
            sev, level, reason, ok, issues = score_one(mdl, tok, desc)
        except Exception as e:
            n_err += 1
            sev, level, reason = None, None, str(e)
            ok, issues = False, [f"exception:{e}"]

        # Adaugam rezultatul in lista
        results.append({
            "image_path": path,
            "disease_category": disease,
            "severity_percent": sev,
            "severity_level": level,
            "severity_reasoning": reason,
            "severity_valid": ok,
            "severity_issues": issues,
        })
        done += 1

        # Salvam la fiecare X pasi pt siguranta
        if done % cfg.save_interval == 0:
            save_out(results)
            tqdm.write(f"  Saved {len(results)} | Done: {done} | Errors: {n_err}")

    # Salvarea finala in afara buclei
    save_out(results)
    return results, n_err

# MAIN

def main():
    random.seed(42)

    print("=" * 70)
    print("  [ARCHIVED] STEP 3: SEVERITY SCORING WITH QWEN2.5-7B-INSTRUCT")
    print("=" * 70)

    # Incarcam setul de date cu prompturile de la MedGemma
    with open(cfg.src_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Filtram pozele cu probleme
    usable = [p for p in raw if not p["generated_prompt"].startswith("ERROR")]
    print(f"  Total images: {len(raw)}")
    print(f"  Usable: {len(usable)}")

    # Calculam o mica estimare de timp
    n_normal = sum(1 for p in usable if p["disease_category"].upper() == "NORMAL")
    n_disease = len(usable) - n_normal
    print(f"  NORMAL (hardcoded 0-10%): {n_normal}")
    print(f"  Disease (Qwen inference): {n_disease}")
    print(f"  Estimated time: ~{n_disease // 60} min on RTX 3090")

    # Lansam executia
    mdl, tok = load_model()
    results, n_err = run_all(mdl, tok, usable)

    # Calculam si printam rezultatele finale in consola
    good = [r for r in results if r.get("severity_valid") is True]
    lvls = [r["severity_level"] for r in good if r.get("severity_level")]
    lvl_dist = dict(sorted(Counter(lvls).items()))

    print("\n" + "=" * 70)
    print(f"  RESULTS:")
    print(f"  Total: {len(results)}")
    print(f"  Valid: {len(good)}")
    print(f"  Errors: {n_err}")
    print(f"  Level distribution: {lvl_dist}")

    # Facem statistica per boala
    for cat in ["NORMAL", "DRUSEN", "AMD", "DME"]:
        subset = [r for r in good if r["disease_category"].upper() == cat]
        if not subset:
            continue

        sevs = [r["severity_percent"] for r in subset if r["severity_percent"] is not None]
        if sevs:
            avg = sum(sevs) / len(sevs)
            print(f"  {cat}: {len(subset)} img | avg: {avg:.1f}% | min: {min(sevs):.1f}% | max: {max(sevs):.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()