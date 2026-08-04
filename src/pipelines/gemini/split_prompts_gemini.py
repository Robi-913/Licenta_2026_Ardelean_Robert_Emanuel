import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from src.utils.seed import set_seed


# CONFIGURARI

class Config:
    api_key = "API_KEY"
    model = "gemini-3.1-flash-lite"

    input_json = "data/OCT5k/medgemma_prompts_v2_27b.json"
    output_json = "data/oct5k/medgemma_prompts_split_v2_27b.json"

    save_interval = 100  # Salvam progresul la fiecare 100 de imagini
    resume = True  # Daca se opreste scriptul, continuam de unde a ramas
    max_retries = 2  # Cate incercari dam daca pica API-ul
    sleep_on_error = 2.0  # Cate secunde asteptam in caz de eroare inainte de retry


cfg = Config()

# Dictionar care mapeaza prescurtarile bolilor pe numele lor intreg, medical
DISEASE_FULL = {
    "AMD": "Age-Related Macular Degeneration",
    "DME": "Diabetic Macular Edema",
    "DRUSEN": "Drusen",
    "NORMAL": "Normal healthy retina",
}

# Instructiunile clare pentru a imparti textul generat in partea de Structura si partea de Patologie
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


def save_results(data):
    # Functie care salveaza dictionarul cu rezultate intr-un fisier JSON
    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_split(response_text, disease_full=""):
    # Functie care extrage cele 2 propozitii din raspunsul primit de la Gemini
    lines = response_text.strip().split("\n")
    prompt_a, prompt_b = "", ""

    for line in lines:
        line = line.strip()
        # Cautam randurile care incep exact cu 'A:' sau 'B:'
        if line.startswith("A:"):
            prompt_a = line[2:].strip()
        elif line.startswith("B:"):
            prompt_b = line[2:].strip()

    # Ne asiguram ca fiecare propozitie incepe cu numele bolii (regula impusa)
    if prompt_a and not prompt_a.startswith(disease_full):
        prompt_a = f"{disease_full} {prompt_a}"
    if prompt_b and not prompt_b.startswith(disease_full):
        prompt_b = f"{disease_full} {prompt_b}"

    # Eliminam punctul de la finalul propozitiei daca AI-ul l-a pus din greseala
    if prompt_a.endswith("."):
        prompt_a = prompt_a[:-1]
    if prompt_b.endswith("."):
        prompt_b = prompt_b[:-1]

    return prompt_a, prompt_b


def main():
    set_seed()

    print("  STEP 3: SPLIT PROMPTS (Structure / Pathology)")
    print(f"  Model: {cfg.model} (no thinking, cost-optimized)")
    print(f"  Input: {cfg.input_json}")

    # Daca cheia este inca placeholder, o cerem din consola (sau o poti seta tu direct)
    if cfg.api_key == "YOUR_API_KEY_HERE" or not cfg.api_key:
        cfg.api_key = input("  Gemini API key: ").strip()

    # Initializam clientul oficial Gemini
    client = genai.Client(api_key=cfg.api_key)

    # Incarcam JSON-ul original creat de MedGemma
    with open(cfg.input_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"  Total prompts: {len(prompts)}")

    # Logica de resume: daca scriptul a rulat deja partial, incarcam munca anterioara
    done = {}
    if cfg.resume and Path(cfg.output_json).exists():
        try:
            with open(cfg.output_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            # Permitem si structura veche (lista) si pe cea noua (dictionar)
            if isinstance(prev, dict):
                done = prev
        except Exception as e:
            print(f"  WARNING resume: {e}")
        print(f"  Resume: {len(done)} already done")

    # Pregatim dictionarul final
    results = dict(done)
    n_new = 0
    n_err = 0

    # Iteram prin poze
    for i, entry in enumerate(tqdm(prompts, desc="  Splitting")):
        img_path = entry["image_path"]
        text = entry.get("generated_prompt", "")
        disease = entry.get("disease_category", "UNKNOWN").upper()
        disease_full = DISEASE_FULL.get(disease, disease)

        # Sarim daca am impartit deja acest text
        if img_path in done:
            continue

        # Tratarea cazurilor in care MedGemma a generat o eroare in pasul precedent
        if not text or text.startswith("ERROR"):
            results[img_path] = {
                "image_path": img_path,
                "a": f"{disease_full} retinal scan with no available description",
                "b": f"{disease_full} retinal scan with no pathological findings described",
            }
            continue

        prompt_a, prompt_b = "", ""

        # Mecanism de Retry in caz ca API-ul Gemini pica temporar
        for attempt in range(cfg.max_retries + 1):
            try:
                # Cererea catre API
                response = client.models.generate_content(
                    model=cfg.model,
                    contents=SPLIT_PROMPT.format(
                        text=text[:1500],  # Limitam la 1500 de caractere pt siguranta
                        disease=disease_full,
                    ),
                    config=types.GenerateContentConfig(
                        max_output_tokens=512,
                        temperature=0.0,  # Vrem rezultate cat mai deterministe
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,  # Modelul e fortat sa NU foloseasca modul de thinking
                        ),
                    ),
                )

                # Reconstructia textului din bucati (daca Gemini a trimis multipart)
                resp_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if part.text:
                        resp_text += part.text

                # Extragem A si B din raspunsul crud
                prompt_a, prompt_b = parse_split(resp_text, disease_full)

                # Daca le-a gasit pe amandoua, spargem bucla de Retry si mergem mai departe
                if prompt_a and prompt_b:
                    break

            except Exception as e:
                # Daca e eroare de retea, asteptam putin inainte de retry
                if attempt < cfg.max_retries:
                    time.sleep(cfg.sleep_on_error)
                else:
                    # Daca a picat de tot, facem un fallback inestetic dar functional taind textul pe jumatate din cod
                    prompt_a = f"{disease_full} " + " ".join(text.split()[:45])
                    prompt_b = f"{disease_full} " + " ".join(text.split()[45:90])
                    n_err += 1

        # Salvam rezultatul pt imaginea curenta
        results[img_path] = {
            "image_path": img_path,
            "a": prompt_a,
            "b": prompt_b,
        }
        n_new += 1

        # Salvare periodica pe disk
        if n_new % cfg.save_interval == 0:
            save_results(results)

    # Salvarea finala in afara buclei
    save_results(results)

    # Calcule statistice finale pentru raportul din consola
    n_good = sum(1 for v in results.values() if v.get("a") and v.get("b"))
    n_empty = sum(1 for v in results.values() if not v.get("a") or not v.get("b"))

    print(f"  DONE!")
    print(f"  Total: {len(results)} | New: {n_new} | Errors: {n_err}")
    print(f"  Good splits: {n_good} | Empty/partial: {n_empty}")
    print(f"\n  Saved: {cfg.output_json}")


if __name__ == "__main__":
    main()
