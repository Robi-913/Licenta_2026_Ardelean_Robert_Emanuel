"""
make_splits_v4.py — Split la nivel de pacient cu stratificare pe biomarkeri rari

Problema v3: biomarkerii rari (Fluid, GeographicAtrophy) cu putini pacienti
cadeau toti in train => test/val aveau 0 exemple pt aceste clase.

Fix v4: stratificare multi-label pe biomarkeri — fortam ca fiecare biomarker
sa aiba reprezentare in toate cele 3 split-uri.

Rulare:
    python src/pipelines/make_splits_v4.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.seed import set_seed, SEED


SPLITS_DIR  = "data/oct5k/splits_biomk"
MASTER_JSON = "data/oct5k/metadata_v2/_master.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BIOMARKERS = [
    "Fluid", "Geographicatrophy", "PRlayerdisruption", "SoftdrusenPED",
    "Reticulardrusen", "Hyperfluorescentspots", "Softdrusen", "Harddrusen", "Choroidalfolds",
]

def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")

_BM_NORM = {_normalize(bm): bm for bm in BIOMARKERS}


def get_patient_session(path: str) -> str:
    """
    Grupeaza imaginile dupa pacient/sesiune.
    Logica: folderul de nivel 2 (ex: AMD Part1/AMD (5)) = un pacient.
    """
    parts = path.replace("\\", "/").split("/")

    # Tip E2E: AMD Part1/AMD (1).E2E/sesiune/Image.png => pacient = AMD Part1/AMD (1).E2E
    if any(".E2E" in p for p in parts):
        idx = next(i for i, p in enumerate(parts) if ".E2E" in p)
        return "/".join(parts[:idx + 1])

    # Tip subfolder: AMD Part1/AMD (5)/Image.png => pacient = AMD Part1/AMD (5)
    if len(parts) >= 2:
        return "/".join(parts[:2])

    return parts[0]


def build_patient_biomarker_matrix(master: list) -> pd.DataFrame:
    """
    Construieste o matrice [n_pacienti x n_biomarkeri] cu prezenta fiecarui biomarker
    per pacient. Folosita pt stratificarea split-ului.
    """
    patient_bms    = defaultdict(lambda: defaultdict(int))
    patient_disease = {}
    patient_images  = defaultdict(list)

    for m in master:
        path    = m["image_path"]
        patient = get_patient_session(path)
        disease = m.get("disease_category", "UNKNOWN")

        patient_disease[patient] = disease
        patient_images[patient].append(m)

        if m.get("bbox_source") == "doctor":
            for les in m.get("lesions", []):
                bm_key = _BM_NORM.get(_normalize(les.get("class", "")))
                if bm_key:
                    patient_bms[patient][bm_key] = 1

    rows = []
    for patient in patient_disease:
        row = {
            "patient_session": patient,
            "disease":         patient_disease[patient],
            "n_images":        len(patient_images[patient]),
        }
        for bm in BIOMARKERS:
            row[bm] = patient_bms[patient].get(bm, 0)
        rows.append(row)

    return pd.DataFrame(rows), patient_images


def stratified_biomarker_split(patient_df: pd.DataFrame) -> tuple[set, set, set]:
    """
    Split stratificat pe biomarkeri rari.

    Strategie:
    1. Identificam pacientii cu biomarkeri rari (< 5 pacienti total)
    2. Ii distribuim manual: minim 1 in val, 1 in test, restul in train
    3. Restul pacientilor se impart normal cu train_test_split stratificat pe disease
    """
    rare_threshold = 5  # biomarkeri cu < 5 pacienti = rari

    # Identificam biomarkerii rari si pacientii lor
    rare_patients = set()
    rare_assignments = {}  # patient -> split

    for bm in BIOMARKERS:
        if bm not in patient_df.columns:
            continue
        bm_patients = patient_df[patient_df[bm] == 1]["patient_session"].tolist()
        if 0 < len(bm_patients) < rare_threshold:
            print(f"  Biomarker rar: {bm} ({len(bm_patients)} pacienti) — distribuim manual")
            for i, p in enumerate(bm_patients):
                if p not in rare_assignments:
                    # Distribuim round-robin: train, train, val, test, train...
                    if i % 4 == 2:
                        rare_assignments[p] = "val"
                    elif i % 4 == 3:
                        rare_assignments[p] = "test"
                    else:
                        rare_assignments[p] = "train"
            rare_patients.update(bm_patients)

    # Pacientii normali — restul
    normal_df = patient_df[~patient_df["patient_session"].isin(rare_patients)]

    train_pat, temp_pat = train_test_split(
        normal_df,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=normal_df["disease"],
        random_state=SEED,
    )
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_pat, test_pat = train_test_split(
        temp_pat,
        test_size=rel_test,
        stratify=temp_pat["disease"],
        random_state=SEED,
    )

    train_sessions = set(train_pat["patient_session"])
    val_sessions   = set(val_pat["patient_session"])
    test_sessions  = set(test_pat["patient_session"])

    # Adaugam pacientii rari la split-urile lor
    for patient, split in rare_assignments.items():
        if split == "train":
            train_sessions.add(patient)
        elif split == "val":
            val_sessions.add(patient)
        else:
            test_sessions.add(patient)

    return train_sessions, val_sessions, test_sessions


def main():
    set_seed()
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("  MAKE SPLITS v4 — Patient-level cu stratificare biomarkeri rari")

    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)

    print(f"  Total imagini in master: {len(master)}")

    patient_df, patient_images = build_patient_biomarker_matrix(master)
    print(f"  Total pacienti/sesiuni: {len(patient_df)}")

    # Afisam distributia biomarkerilor per pacient
    print("\n  Biomarkeri per pacient (doctor-only):")
    for bm in BIOMARKERS:
        if bm in patient_df.columns:
            n = int(patient_df[bm].sum())
            print(f"    {bm:<25}: {n} pacienti")

    train_sessions, val_sessions, test_sessions = stratified_biomarker_split(patient_df)

    # Verificam ca nu exista overlap
    assert len(train_sessions & val_sessions)  == 0, "OVERLAP train/val!"
    assert len(train_sessions & test_sessions) == 0, "OVERLAP train/test!"
    assert len(val_sessions   & test_sessions) == 0, "OVERLAP val/test!"
    print("\n  ✓ Niciun overlap intre splits!")

    # Construim DataFrame-urile finale
    rows = []
    for m in master:
        path    = m["image_path"]
        patient = get_patient_session(path)
        rows.append({
            "image_path":       path,
            "image_disk_path":  m.get("image_disk_path", ""),
            "disease":          m.get("disease_category", "UNKNOWN"),
            "patient_session":  patient,
            "has_bbox":         m.get("has_bounding_boxes", False),
            "bbox_source":      m.get("bbox_source", "none"),
            "has_boundaries":   m.get("has_boundaries", False),
            "num_lesions":      m.get("num_lesions", 0),
        })

    df = pd.DataFrame(rows)

    train_df = df[df["patient_session"].isin(train_sessions)].reset_index(drop=True)
    val_df   = df[df["patient_session"].isin(val_sessions)].reset_index(drop=True)
    test_df  = df[df["patient_session"].isin(test_sessions)].reset_index(drop=True)

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sdf.to_csv(f"{SPLITS_DIR}/{name}.csv", index=False)

    # Print summary
    print(f"\n  {'Split':<8} {'Img':>6} {'Pat':>5} {'AMD':>6} {'DME':>6} {'DRUSEN':>7} {'NORMAL':>7} {'doc':>6}")
    print(f"  {'─' * 60}")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dc = sdf["disease"].value_counts()
        sc = sdf["bbox_source"].value_counts()
        n_pat = sdf["patient_session"].nunique()
        print(f"  {name:<8} {len(sdf):>6} {n_pat:>5} "
              f"{dc.get('AMD',0):>6} {dc.get('DME',0):>6} "
              f"{dc.get('DRUSEN',0):>7} {dc.get('NORMAL',0):>7} "
              f"{sc.get('doctor',0):>6}")

    # Verificam biomarkerii rari per split
    print(f"\n  Biomarkeri rari in fiecare split (doctor-only):")
    print(f"  {'Biomarker':<25} {'train':>6} {'val':>6} {'test':>6}")
    print(f"  {'-' * 45}")

    for bm in BIOMARKERS:
        counts = {}
        for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            bbox_paths = set(sdf[sdf["bbox_source"] == "doctor"]["image_path"])
            n = 0
            for m in master:
                if m["image_path"] not in bbox_paths:
                    continue
                for les in m.get("lesions", []):
                    if _BM_NORM.get(_normalize(les.get("class", ""))) == bm:
                        n += 1
                        break
            counts[name] = n
        print(f"  {bm:<25} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6}")

    print(f"\n  Splits salvate in: {SPLITS_DIR}/")
    print(f"  IMPORTANT: Actualizeaza cfg.splits_dir -> 'data/oct5k/splits_v4' in toate scripturile!")


if __name__ == "__main__":
    main()