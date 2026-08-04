"""
Logica fluxului:
  - Doctor -> Calcul determinist direct bazat pe adnotarile sigure (ground truth).
  - YOLO   -> MedGemma primeste predictiile YOLO, verifica imaginea, modifica numarul de leziuni, apoi se face calculul determinist pe lista corectata.
  - None   -> MedGemma analizeaza imaginea de la zero, numara leziunile, apoi calcul determinist.
"""

import gc
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


class Config:
    model_path = "model/medgemma-27b-it"
    load_in_4bit = True

    master_json = "data/oct5k/metadata_v2/_master.json"
    out_json = "data/oct5k/severity_scores_v2.json"

    max_tokens = 256
    retries = 2  # Cate incercari dam modelului daca nu respecta formatul
    save_interval = 50
    resume = True
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

# Importanta fiecarei leziuni individuale adaugata la scorul de severitate
BIOMARKER_WEIGHTS = {
    "Fluid": 0.30,
    "Geographicatrophy": 0.25,
    "PRlayerdisruption": 0.20,
    "SoftdrusenPED": 0.12,
    "Reticulardrusen": 0.07,
    "Hyperfluorescentspots": 0.06,
    "Softdrusen": 0.04,
    "Harddrusen": 0.02,
    "Choroidalfolds": 0.01,
}

# Un scor minim garantat doar pentru simplul fapt ca ochiul are o anumita boala
DISEASE_BASE_SCORE = {
    "AMD": 0.15,
    "DME": 0.20,
    "DRUSEN": 0.10,
    "NORMAL": 0.02,
}

AREA_WEIGHT = 0.05
COUNT_WEIGHT = 0.15
MAX_BIOMARKER_COUNT = 4


def _count_bonus(n):
    # Daca o leziune apare de 100 de ori, nu vrem ca scorul sa explodeze peste 100%
    # Folosim o functie logaritmica pt a plafona bonusul de numar la maxim 'COUNT_WEIGHT' (15%)
    n_capped = min(n, MAX_BIOMARKER_COUNT)
    if n_capped == 0:
        return 0.0
    return math.log(n_capped + 1) / math.log(MAX_BIOMARKER_COUNT + 1) * COUNT_WEIGHT


def normalize_class(cls):
    # Transforma "Hard Drusen", "hard_drusen" si "Harddrusen" in aceeasi cheie pentru a nu rata potriviri
    return cls.lower().replace(" ", "").replace("_", "")


# Un dictionar de mapare inversa (ex: 'harddrusen' -> 'Harddrusen')
BIOMARKER_KEYS_NORMALIZED = {normalize_class(k): k for k in BIOMARKER_WEIGHTS}


def get_level(pct):
    # Transforma un procentaj intr-o categorie clinica usor de citit de oameni
    if pct < 15:  return "Minimal"
    if pct < 30:  return "Mild"
    if pct < 50:  return "Moderate"
    if pct < 70:  return "Significant"
    if pct < 85:  return "Severe"
    return "Critical"


def compute_deterministic(biomarker_list, disease, area_pct=0.0):
    # Calculeaza matematic scorul pe baza unei liste clare de leziuni.
    # Ex intrare: ["Harddrusen", "Harddrusen", "Softdrusen"]
    base = DISEASE_BASE_SCORE.get(disease.upper(), 0.02)

    biomarker_score = 0.0
    for bm in biomarker_list:
        weight = BIOMARKER_WEIGHTS.get(bm, 0)
        biomarker_score += weight

    counts = Counter(biomarker_list)
    area_score = (area_pct / 100.0) * AREA_WEIGHT
    count_score = _count_bonus(len(biomarker_list))

    raw = base + biomarker_score + area_score + count_score
    # Ne asiguram ca scorul nu depaseste 100 si nu e sub 0
    sev = round(min(100.0, max(0.0, raw * 100)), 1)

    return sev, {
        "base_score": round(base * 100, 1),
        "biomarker_score": round(biomarker_score * 100, 1),
        "area_score": round(area_score * 100, 1),
        "count_bonus": round(count_score * 100, 1),
        "biomarker_counts": dict(counts),
        "total_instances": len(biomarker_list),
    }


IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


def locate_image(meta):
    disk = meta.get("image_disk_path", "")
    if disk and Path(disk).exists():
        return str(disk)
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in IMG_DIRS:
        full = Path(base) / rel
        if full.exists():
            return str(full)
        for ext in [".png", ".jpeg", ".jpg"]:
            if full.with_suffix(ext).exists():
                return str(full.with_suffix(ext))
    return None


def load_model():
    print(f"\n  Model: {cfg.model_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    proc = AutoProcessor.from_pretrained(cfg.model_path)
    mdl = AutoModelForImageTextToText.from_pretrained(
        cfg.model_path,
        quantization_config=quant_cfg,
        device_map="auto",
        max_memory={0: "23GiB", "cpu": "26GiB"},
        low_cpu_mem_usage=True,
        offload_folder="offload_tmp",
        torch_dtype=torch.bfloat16,
    )
    mdl.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM: {used:.1f}GB / {total:.1f}GB")
    print("  Loaded!\n")
    return mdl, proc


SEVERITY_PROMPT = """\
You are an expert ophthalmologist analyzing a retinal OCT scan.

Look carefully at this OCT image and identify ALL biomarkers visible, counting each instance.

Biomarker weights per instance:
- Fluid (IRF/SRF): +30% each
- Geographic Atrophy (RPE loss): +25% each
- PR Layer Disruption (EZ/ELM disruption): +20% each
- Soft Drusen PED (large RPE elevation >350um): +12% each
- Reticular Drusen (subretinal deposits above RPE): +7% each
- Hyperreflective Foci (small bright dots): +6% each
- Soft Drusen (dome-shaped RPE elevations): +4% each
- Hard Drusen (small discrete deposits <63um): +2% each
- Choroidal Folds (undulating RPE lines): +1% each

Disease category: {disease} (base score: {base_score}%)
{yolo_hint}
Instructions:
1. Count EACH individual instance of every biomarker you can see
2. Report the count per biomarker type
3. Calculate: base + sum(weight x count for each biomarker)

Output EXACTLY in this format (3 lines only):
Present: <biomarker x count, e.g. "Harddrusen x4, Softdrusen x2", or "none">
Reasoning: <1 sentence about what you see and how many>
Severity: <number>%"""

YOLO_HINT_TEMPLATE = """
An automated YOLO detector found these biomarkers (verify and correct the counts):
{biomarkers}
Please check each one: confirm, correct the count, remove false positives, and add any biomarkers YOLO missed.
"""


def format_yolo_hints(lesions):
    # Formateaza predictiile brute YOLO intr-un string citibil pt prompt
    counts = Counter(l["class"] for l in lesions)
    return ", ".join(f"{bm} x{n}" for bm, n in counts.most_common())


@torch.no_grad()
def call_medgemma(mdl, proc, image, disease, yolo_hints_str=None, extra=""):
    # Construieste promptul si il trimite catre model
    base_score = int(DISEASE_BASE_SCORE.get(disease.upper(), 0.02) * 100)

    hint_text = ""
    if yolo_hints_str:
        hint_text = YOLO_HINT_TEMPLATE.format(biomarkers=yolo_hints_str)

    prompt = SEVERITY_PROMPT.format(
        disease=disease,
        base_score=base_score,
        yolo_hint=hint_text,
    ) + extra

    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )
    prefix_len = inputs["input_ids"].shape[1]

    feed = {
        k: inputs[k].to(mdl.device)
        for k in ["input_ids", "attention_mask", "pixel_values"]
        if k in inputs
    }

    out = mdl.generate(
        **feed,
        max_new_tokens=cfg.max_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=proc.tokenizer.eos_token_id,
    )

    return proc.tokenizer.decode(out[0][prefix_len:], skip_special_tokens=True).strip()


def parse_medgemma_response(response, disease):
    # Extrage componentele utile din textul generat de AI
    sev_from_text = None
    present_line = ""
    reasoning = ""

    for line in response.split("\n"):
        line = line.strip()
        low = line.lower()

        if low.startswith("severity:"):
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if m:
                sev_from_text = max(0, min(100, float(m.group(1))))
        elif low.startswith("present:"):
            present_line = line.split(":", 1)[1].strip()
        elif low.startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    # Expandeaza linia de tipul "Harddrusen x4" intr-o lista [HD, HD, HD, HD]
    expanded = []
    if present_line and present_line.lower() not in ["none", "n/a", "-", ""]:
        for part in present_line.split(","):
            part = part.strip()

            # Extrage cuvantul si multiplicatorul (xN)
            m = re.match(r"(.+?)\s*x\s*(\d+)", part, re.IGNORECASE)
            if m:
                bm_text = m.group(1).strip()
                count = int(m.group(2))
            else:
                bm_text = part
                count = 1

            # Mapeaza pe numele oficial ca sa nu existe greseli de scriere
            key = BIOMARKER_KEYS_NORMALIZED.get(normalize_class(bm_text))
            if key:
                expanded.extend([key] * count)

    return expanded, reasoning, sev_from_text


def score_with_medgemma(mdl, proc, image, disease, yolo_hints_str=None):
    # Incearca sa obtina un scor corect, aplicand retry daca modelul halucineaza un format gresit
    for attempt in range(cfg.retries + 1):
        extra = ""
        if attempt > 0:
            extra = "\n\nRetry: Use format 'Present: Biomarker xN, ...' with exact counts."

        try:
            raw = call_medgemma(mdl, proc, image, disease, yolo_hints_str, extra)
        except Exception as e:
            return [], str(e), None, False, [f"error:{e}"]

        expanded, reasoning, sev_raw = parse_medgemma_response(raw, disease)

        # Succes complet (avem numere de leziuni)
        if expanded:
            return expanded, reasoning, sev_raw, True, []

        # Succes partial (n-a stiut sa numere dar a dat direct un procentaj estimativ)
        if sev_raw is not None:
            return [], reasoning, sev_raw, True, ["no_counts_fallback_sev"]

    # Daca a esuat si dupa retries
    return [], "parse_failed", None, False, ["parse_failed"]


def save_out(data):
    out = Path(cfg.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_all(metadata):
    # Incarca progresul vechi pt a preveni recalcularea inutila
    prev = {}
    if cfg.resume and Path(cfg.out_json).exists():
        with open(cfg.out_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        prev = {x["image_path"]: x for x in old if x.get("severity_valid") is True}
        print(f"  Resume: {len(prev)} already done")

    results = list(prev.values())
    done_det = 0
    done_mg = 0
    n_err = 0

    # Impartim pozele pe categorii de sursa pt a sti cum le tratam
    doctor_bbox = [m for m in metadata if m.get("bbox_source") == "doctor"]
    yolo_bbox = [m for m in metadata if m.get("bbox_source") == "yolo"]
    no_bbox = [m for m in metadata if m.get("bbox_source") == "none" or not m.get("bbox_source")]

    print(f"  doctor (deterministic):  {len(doctor_bbox)}")
    print(f"  yolo (MedGemma verify):  {len(yolo_bbox)}")
    print(f"  none (MedGemma direct):  {len(no_bbox)}")

    # PASUL 1: Doctor -> Scorul este strict matematic
    print("\n  [1/2] Doctor bbox - deterministic...")
    for meta in tqdm(doctor_bbox, desc="  Doctor"):
        path = meta["image_path"]
        if path in prev:
            continue

        # Convertim dintr-un array de dict-uri intr-o lista simpla de aparitii [HD, HD, SD]
        bm_list = [l["class"] for l in meta.get("lesions", [])
                   if BIOMARKER_KEYS_NORMALIZED.get(normalize_class(l.get("class", "")))]
        bm_list = [BIOMARKER_KEYS_NORMALIZED[normalize_class(b)] for b in bm_list]

        sev, breakdown = compute_deterministic(
            bm_list,
            meta["disease_category"],
            meta.get("total_lesion_area_percent", 0),
        )

        results.append({
            "image_path": path,
            "disease_category": meta["disease_category"],
            "severity_percent": sev,
            "severity_level": get_level(sev),
            "severity_valid": True,
            "severity_method": "deterministic_doctor",
            "severity_issues": [],
            "breakdown": breakdown,
        })
        done_det += 1

    # PASUL 2: YOLO si Cele Fara Cutii -> Intervine MedGemma
    needs_mg = [m for m in (yolo_bbox + no_bbox) if m["image_path"] not in prev]

    if not needs_mg:
        print("\n  Toate deja procesate!")
    else:
        print(f"\n  [2/2] MedGemma pentru {len(needs_mg)} imagini...")
        mdl, proc = load_model()

        for meta in tqdm(needs_mg, desc="  MedGemma"):
            path = meta["image_path"]
            disease = meta["disease_category"]

            if path in prev:
                continue

            # Bypass pt ochii sanatosi. Atribuim o severitate complet normala, mica si randomizata putin
            if disease.upper() == "NORMAL":
                sev = round(random.uniform(0, 8), 1)
                results.append({
                    "image_path": path,
                    "disease_category": disease,
                    "severity_percent": sev,
                    "severity_level": get_level(sev),
                    "severity_valid": True,
                    "severity_method": "hardcoded_normal",
                    "severity_issues": [],
                    "breakdown": {"base_score": sev, "biomarker_counts": {}, "total_instances": 0},
                })
                continue

            # Incarca imaginea fizica pt AI
            img_file = locate_image(meta)
            if img_file is None:
                results.append({
                    "image_path": path, "disease_category": disease,
                    "severity_percent": None, "severity_valid": False,
                    "severity_method": "medgemma", "severity_issues": ["image_not_found"],
                })
                continue

            try:
                img = Image.open(img_file).convert("RGB")
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            except Exception as e:
                results.append({
                    "image_path": path, "disease_category": disease,
                    "severity_percent": None, "severity_valid": False,
                    "severity_method": "medgemma", "severity_issues": [f"load_error:{e}"],
                })
                n_err += 1
                continue

            # Pregateste hint-urile YOLO pt prompt (Daca exista)
            lesions = meta.get("lesions", [])
            bbox_src = meta.get("bbox_source", "none")

            yolo_hints_str = None
            yolo_counts = {}
            if lesions and bbox_src == "yolo":
                yolo_hints_str = format_yolo_hints(lesions)
                yolo_counts = dict(Counter(l["class"] for l in lesions))

            # Rularea efectiva a retelei MedGemma
            expanded, reasoning, sev_raw, ok, issues = score_with_medgemma(
                mdl, proc, img, disease, yolo_hints_str
            )

            if not ok:
                n_err += 1

            # Acum ca MedGemma a dat lista corecta de leziuni, putem rula scorul matematic pe ele
            if expanded:
                sev, breakdown = compute_deterministic(
                    expanded, disease,
                    meta.get("total_lesion_area_percent", 0),
                )
                method = "deterministic_medgemma_yolo" if bbox_src == "yolo" else "deterministic_medgemma"
            elif sev_raw is not None:
                sev = sev_raw
                breakdown = None
                method = "medgemma_fallback"
            else:
                sev = DISEASE_BASE_SCORE.get(disease.upper(), 0.02) * 100
                breakdown = None
                method = "base_score_only"

            mg_counts = dict(Counter(expanded))

            # Impachetam rezultatul imaginii curente
            entry = {
                "image_path": path,
                "disease_category": disease,
                "severity_percent": round(sev, 1) if sev else None,
                "severity_level": get_level(sev) if sev else None,
                "severity_valid": ok,
                "severity_method": method,
                "severity_issues": issues,
                "reasoning": reasoning,
                "medgemma_counts": mg_counts,
            }

            # Pastram statistici despre cat a schimbat MedGemma fata de ce zicea YOLO initial
            if yolo_counts:
                entry["yolo_counts"] = yolo_counts
                entry["comparison"] = {
                    "yolo_total": sum(yolo_counts.values()),
                    "medgemma_total": sum(mg_counts.values()),
                    "added_by_medgemma": list(set(mg_counts.keys()) - set(yolo_counts.keys())),
                    "removed_by_medgemma": list(set(yolo_counts.keys()) - set(mg_counts.keys())),
                }

            if breakdown:
                entry["breakdown"] = breakdown

            results.append(entry)
            done_mg += 1

            if done_mg % cfg.save_interval == 0:
                save_out(results)
                tqdm.write(f"  MedGemma: {done_mg} | Errors: {n_err}")

    # Salvare finala si iesire
    save_out(results)
    return results, done_det, done_mg, n_err


def main():
    random.seed(42)

    print("  SEVERITY SCORING v2")
    print("  doctor: deterministic cu counts")
    print("  yolo:   MedGemma verifica + corecteaza counts -> deterministic")
    print("  none:   MedGemma detecteaza cu counts -> deterministic")

    with open(cfg.master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Total imagini: {len(metadata)}")

    results, done_det, done_mg, n_err = run_all(metadata)

    # Statistici finale post-procesare
    good = [r for r in results if r.get("severity_valid") is True]
    stats = defaultdict(list)
    method_counts = Counter(r.get("severity_method") for r in good)

    for r in good:
        stats[r["disease_category"].upper()].append(r["severity_percent"])

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {len(results)} total | {len(good)} valid | {n_err} errors")
    print(f"\n  Methods:")
    for method, count in method_counts.items():
        print(f"    {method}: {count}")

    print(f"\n  Per disease:")
    for cat in ["NORMAL", "DRUSEN", "AMD", "DME"]:
        sevs = stats.get(cat, [])
        if sevs:
            print(f"  {cat}: {len(sevs)} | avg={sum(sevs) / len(sevs):.1f}% | "
                  f"min={min(sevs):.1f}% | max={max(sevs):.1f}%")

    # Comparatie : YOLO vs MedGemma
    yolo_entries = [r for r in good if r.get("yolo_counts")]
    if yolo_entries:
        n_added = sum(len(r.get("comparison", {}).get("added_by_medgemma", [])) for r in yolo_entries)
        n_removed = sum(len(r.get("comparison", {}).get("removed_by_medgemma", [])) for r in yolo_entries)
        print(f"\n  YOLO vs MedGemma comparison ({len(yolo_entries)} images):")
        print(f"    Biomarker types added by MedGemma:   {n_added}")
        print(f"    Biomarker types removed by MedGemma: {n_removed}")

    print(f"\n  Saved: {cfg.out_json}")


if __name__ == "__main__":
    main()
