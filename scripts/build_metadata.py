import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.seed import set_seed, SEED


class Config:
    oct5k_root = "data/OCT5k"

    bb_csv = "data/OCT5k/Detection/all_bounding_boxes.csv"
    classes_csv = "data/OCT5k/Detection/all_classes.csv"

    img_dirs = [
        "data/OCT5k/Images/Images_Automatic",
        "data/OCT5k/Images/Images_Manual",
        "data/OCT5k/Detection/Images",
    ]

    bounds_auto = "data/OCT5k/Boundaries/Boundaries_Automatic/Grading"
    bounds_manual = "data/OCT5k/Boundaries/Boundaries_Manual/Grading_1"

    masks_auto = "data/OCT5k/Masks/Masks_Automatic_RGB/Grading"
    masks_manual = "data/OCT5k/Masks/Masks_Manual_RGB/Grading_1"

    meta_dir = "data/oct5k/metadata_v2"
    splits_dir = "data/oct5k/splits_v3"  # v3 = split la nivel de pacient

    yolo_bboxes_json = "data/oct5k/yolo_bboxes.json"

    img_size = 512

    # Procentajele pt split-ul datelor
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Mapam numele folderelor la niste categorii standardizate de boli
    disease_map = {
        "AMD Part1": "AMD",
        "AMD Part2": "AMD",
        "DME": "DME",
        "DRUSEN": "DRUSEN",
        "Normal Part1": "NORMAL",
        "Normal Part2": "NORMAL",
    }

    # Straturile retinei care ne intereseaza
    layers = ["ILM", "OPL", "IS-OS", "IBRPE", "OBRPE"]

    # Definim regiunile dintre straturi
    regions = {
        "RNFL_GCL_IPL": ("ILM", "OPL"),
        "INL_OPL": ("OPL", "IS-OS"),
        "photoreceptors": ("IS-OS", "IBRPE"),
        "RPE": ("IBRPE", "OBRPE"),
    }


cfg = Config()


def get_disease(path):
    # Extragem tipul bolii din numele primului folder (ex: 'DME/Image.png' -> 'DME')
    normalized = path.replace("\\", "/")
    folder = normalized.split("/")[0]
    return cfg.disease_map.get(folder, "UNKNOWN")


def get_patient_session(path):
    # Functie care izoleaza id-ul unic al pacientului pt a preveni data leakage la split
    parts = path.replace("\\", "/").split("/")

    # Tip 1: fisiere .E2E (ex: AMD Part1/AMD (7).E2E/2-25-2017/Image.png)
    if any(".E2E" in p for p in parts):
        idx = next(i for i, p in enumerate(parts) if ".E2E" in p)
        return "/".join(parts[:idx + 2])

    # Tip 2: non-E2E cu subfolder (ex: AMD Part1/AMD (21)/Image (13).png)
    if len(parts) >= 3 and parts[-1].lower().endswith(".png"):
        if not parts[1].endswith(".png"):
            return "/".join(parts[:2])

    # Tip 3: structura flat (ex: DRUSEN/DRUSEN-9884539-8.png)
    if len(parts) == 2:
        fname = parts[-1].replace(".png", "").replace(".jpeg", "")
        patient_id = "-".join(fname.split("-")[:-1])
        return f"{parts[0]}/{patient_id}" if patient_id else f"{parts[0]}/{fname}"

    # Fallback
    return "/".join(parts[:2])


def make_key(path):
    # Generam un ID unic (hash) per imagine pentru a nu avea fisiere cu acelasi nume
    h = hashlib.md5(path.encode()).hexdigest()[:10]
    clean = path.replace("\\", "_").replace("/", "_").replace(" ", "_")
    clean = clean.replace("(", "").replace(")", "").replace(".", "_")
    clean = clean.replace("_png", "").replace("_jpeg", "").replace("_PNG", "")

    # Curatam underscore-urile multiple
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")

    if len(clean) > 60:
        clean = clean[:60]
    return f"{clean}_{h}"


def find_image(rel_path):
    # Cautam imaginea in cele 3 foldere definite in config
    for base in cfg.img_dirs:
        full = os.path.join(base, rel_path)
        if os.path.exists(full):
            return full
        # Daca nu o gasim fix sub acest nume, incercam si alte extensii
        for ext in [".png", ".jpeg", ".jpg"]:
            alt = os.path.splitext(full)[0] + ext
            if os.path.exists(alt):
                return alt
    return None


def find_boundary(rel_path):
    # Inlocuim extensia de imagine cu .csv si cautam fisierul de delimitari (boundaries)
    csv_rel = os.path.splitext(rel_path.replace("\\", "/"))[0] + ".csv"
    for base in [cfg.bounds_auto, cfg.bounds_manual]:
        full = os.path.join(base, csv_rel)
        if os.path.exists(full):
            return full
    return None


def find_mask(rel_path):
    # Similar, inlocuim extensia si cautam masca (segmentarea pixel-perfect)
    png_rel = os.path.splitext(rel_path.replace("\\", "/"))[0] + ".png"
    for base in [cfg.masks_auto, cfg.masks_manual]:
        full = os.path.join(base, png_rel)
        if os.path.exists(full):
            return full
    return None


def retinal_zone(cx_norm):
    # Imparte imaginea in 3 zone verticale (0-33%, 33-66%, 66-100%)
    if cx_norm < 0.33:
        return "nasal"
    if cx_norm < 0.66:
        return "central-foveal"
    return "temporal"


def parse_boundaries(csv_path):
    # Citim limitele straturilor. Daca lipseste ceva sau fisierul e corupt, anulam (return None)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if not all(col in df.columns for col in cfg.layers):
        return None

    # Calculam statistici (medie, deviatie, min, max) pt inaltimea fiecarui strat in parte
    layer_stats = {}
    for name in cfg.layers:
        vals = df[name].values.astype(float)
        layer_stats[name] = {
            "mean_y": round(float(np.mean(vals)), 1),
            "std_y": round(float(np.std(vals)), 2),
            "min_y": int(np.min(vals)),
            "max_y": int(np.max(vals)),
        }

    # Calculam grosimea fiecarei regiuni importante
    region_stats = {}
    for rname, (top, bot) in cfg.regions.items():
        thick = df[bot].values.astype(float) - df[top].values.astype(float)
        region_stats[rname] = {
            "mean_thickness_px": round(float(np.mean(thick)), 1),
            "std_thickness_px": round(float(np.std(thick)), 2),
            "min_thickness_px": int(np.min(thick)),
            "max_thickness_px": int(np.max(thick)),
            "mean_thickness_pct": round(float(np.mean(thick)) / cfg.img_size * 100, 2),
        }

    # Calculam grosimea totala a retinei (de la primul la ultimul strat)
    total = df["OBRPE"].values.astype(float) - df["ILM"].values.astype(float)
    t_mean, t_std = np.mean(total), np.std(total)

    # Identificam anomaliile: unde retina e mult prea groasa/subtire (deformari) fata de media ochiului
    deformations = []
    if t_std > 3:
        for x in range(len(total)):
            # Daca deviatia intr-un pixel e mai mare de 2 deviatii standard, o inregistram
            if abs(total[x] - t_mean) > 2 * t_std:
                deformations.append({
                    "x_position": x,
                    "x_normalized": round(x / cfg.img_size, 3),
                    "zone": retinal_zone(x / cfg.img_size),
                    "thickness_px": int(total[x]),
                    "deviation_from_mean_px": round(float(total[x] - t_mean), 1),
                    "type": "thickening" if total[x] > t_mean else "thinning",
                })

    return {
        "layers": layer_stats,
        "regions": region_stats,
        "total_retinal_thickness": {
            "mean_px": round(float(t_mean), 1),
            "std_px": round(float(t_std), 2),
            "min_px": int(np.min(total)),
            "max_px": int(np.max(total)),
            "mean_pct": round(float(t_mean) / cfg.img_size * 100, 2),
        },
        "deformation_zones": deformations[:10],  # Pastram doar primele 10 pt a nu aglomera json-ul
        "num_deformations": len(deformations),
    }


def correlate_bbox_layers(bbox, bounds):
    # Estimeaza in ce strat se afla centrul cutiei (bounding box-ului) detectate
    if bounds is None:
        return {"affected_layer": "unknown", "depth_info": "no boundary data"}

    xmin, ymin, xmax, ymax = bbox
    cy = (ymin + ymax) // 2

    # Ordonam straturile pe axa Y (de sus in jos)
    layer_y = {name: bounds["layers"][name]["mean_y"] for name in cfg.layers}
    ordered = sorted(layer_y.items(), key=lambda x: x[1])

    # Cazul 1: Cutia e deasupra primului strat (in umoarea vitroasa)
    if cy < ordered[0][1]:
        return {
            "affected_layer": "above_ILM",
            "depth_info": "vitreous space, above inner limiting membrane",
            "closest_layer": ordered[0][0],
            "distance_to_closest_px": round(ordered[0][1] - cy, 1),
        }

    # Cazul 2: Cutia e sub ultimul strat (coroida)
    if cy > ordered[-1][1]:
        return {
            "affected_layer": "below_OBRPE",
            "depth_info": "choroidal space, below outer Bruch's RPE",
            "closest_layer": ordered[-1][0],
            "distance_to_closest_px": round(cy - ordered[-1][1], 1),
        }

    # Cazul 3: Cutia se afla intre doua straturi
    for i in range(len(ordered) - 1):
        top_name, top_y = ordered[i]
        bot_name, bot_y = ordered[i + 1]

        if top_y <= cy <= bot_y:
            region = "unknown"
            for rname, (rtop, rbot) in cfg.regions.items():
                if rtop == top_name and rbot == bot_name:
                    region = rname
                    break

            rel_depth = (cy - top_y) / max(1, bot_y - top_y)
            label = region if region != "unknown" else f"between_{top_name}_and_{bot_name}"
            return {
                "affected_layer": label,
                "between_layers": [top_name, bot_name],
                "relative_depth_in_region": round(rel_depth, 3),
                "depth_info": f"located between {top_name} (y={top_y:.0f}) and {bot_name} (y={bot_y:.0f})",
            }

    return {"affected_layer": "unknown", "depth_info": "could not determine"}


def _build_lesion_entry(xmin, ymin, xmax, ymax, cls_name, bounds, conf=None):
    xmin_n = round(xmin / cfg.img_size, 4)
    ymin_n = round(ymin / cfg.img_size, 4)
    xmax_n = round(xmax / cfg.img_size, 4)
    ymax_n = round(ymax / cfg.img_size, 4)

    cx_n = (xmin_n + xmax_n) / 2
    cy_n = (ymin_n + ymax_n) / 2

    area_px = (xmax - xmin) * (ymax - ymin)
    area_pct = round(100.0 * area_px / (cfg.img_size ** 2), 2)

    layer_info = correlate_bbox_layers((xmin, ymin, xmax, ymax), bounds)

    lesion = {
        "class": cls_name,
        "bbox_px": [xmin, ymin, xmax, ymax],
        "bbox_normalized": [xmin_n, ymin_n, xmax_n, ymax_n],
        "center_normalized": [round(cx_n, 4), round(cy_n, 4)],
        "size_px": [xmax - xmin, ymax - ymin],
        "area_percent": area_pct,
        "retinal_zone": retinal_zone(cx_n),
        "layer_correlation": layer_info,
    }

    # Daca a venit de la YOLO, salvam si cat de sigura a fost predictia
    if conf is not None:
        lesion["yolo_confidence"] = round(conf, 4)

    return lesion


def process_bboxes(img_path, bb_group, bounds):
    # Procesam datele marcate manual de doctori
    return [
        _build_lesion_entry(
            int(row["xmin"]), int(row["ymin"]),
            int(row["xmax"]), int(row["ymax"]),
            row["class"], bounds
        )
        for _, row in bb_group.iterrows()
    ]


def process_yolo_bboxes(yolo_lesions, bounds):
    # Procesam la fel si detectiile YOLO
    return [
        _build_lesion_entry(
            int(les["bbox_abs"][0]), int(les["bbox_abs"][1]),
            int(les["bbox_abs"][2]), int(les["bbox_abs"][3]),
            les["class"], bounds, les.get("confidence", 0.0)
        )
        for les in yolo_lesions
    ]


def collect_images():
    # Parcurge folderele fizice si returneaza un dictionar cu pozele gasite
    found = {}

    for base_dir in cfg.img_dirs:
        if not os.path.exists(base_dir):
            print(f"  WARNING: {base_dir} does not exist")
            continue

        for root, _, files in os.walk(base_dir):
            for fname in files:
                if not fname.lower().endswith((".png", ".jpeg", ".jpg")):
                    continue

                full = os.path.join(root, fname)
                rel = os.path.relpath(full, base_dir)
                base_rel = os.path.splitext(rel)[0]

                # Evitam dublurile prioritizand PNG-urile
                if base_rel not in found or rel.endswith(".png"):
                    found[base_rel] = {
                        "rel_path": rel,
                        "disk_path": full,
                        "source": os.path.basename(base_dir),
                    }

    return found


def build_metadata():
    # Construim fisierul urias de metadate ce centralizeaza absolut tot per imagine
    print("  STEP 1: BUILD STRUCTURED METADATA FOR MEDGEMMA")

    os.makedirs(cfg.meta_dir, exist_ok=True)
    os.makedirs(cfg.splits_dir, exist_ok=True)

    bb_df = pd.read_csv(cfg.bb_csv)
    bb_grouped = dict(list(bb_df.groupby("image")))
    print(f"\n  Bounding boxes doctor: {len(bb_df)} total, {len(bb_grouped)} imagini")

    yolo_bboxes = {}
    if os.path.exists(cfg.yolo_bboxes_json):
        with open(cfg.yolo_bboxes_json, "r", encoding="utf-8") as f:
            yolo_bboxes_raw = json.load(f)
            yolo_bboxes = {k.replace("\\", "/"): v for k, v in yolo_bboxes_raw.items()}
        n_yolo_with_det = sum(
            1 for v in yolo_bboxes.values()
            if v["bbox_source"] == "yolo" and len(v.get("lesions", [])) > 0
        )
        print(f"  YOLO bboxes: {len(yolo_bboxes)} imagini, {n_yolo_with_det} cu detectii")
    else:
        print(f"  WARNING: {cfg.yolo_bboxes_json} not found - YOLO bboxes disabled")

    images = collect_images()
    print(f"  Images found: {len(images)}")

    all_meta = []
    counts = defaultdict(int)

    for base_rel, info in sorted(images.items()):
        rel = info["rel_path"]
        disk = info["disk_path"]
        disease = get_disease(rel)

        # Cautam adnotari externe (limite straturi si masti RGB)
        bound_path = find_boundary(rel)
        bounds = parse_boundaries(bound_path) if bound_path else None
        if bounds:
            counts["has_bounds"] += 1

        mask_path = find_mask(rel)
        if mask_path:
            counts["has_mask"] += 1

        lesions = []
        bbox_source = "none"

        rel_norm = rel.replace("\\", "/")
        # Ne asiguram ca gasim bb-ul doctorului indiferent de extensie
        candidates = [
            rel_norm,
            rel_norm.replace(".jpeg", ".png"),
            rel_norm.replace(".jpg", ".png"),
        ]

        bb_key = next((c for c in candidates if c in bb_grouped), None)

        # Daca avem cutii desenate de doctor, le punem. Daca nu, dar are YOLO, le punem pe alea
        if bb_key:
            lesions = process_bboxes(rel, bb_grouped[bb_key], bounds)
            bbox_source = "doctor"
            counts["has_bbox_doctor"] += 1
        elif rel_norm in yolo_bboxes:
            yolo_entry = yolo_bboxes[rel_norm]
            yolo_lesions = yolo_entry.get("lesions", [])
            if yolo_lesions:
                lesions = process_yolo_bboxes(yolo_lesions, bounds)
                bbox_source = "yolo"
                counts["has_bbox_yolo"] += 1

        # Centralizam dict-ul final de metadata
        meta = {
            "image_path": rel,
            "image_disk_path": disk,
            "disease_category": disease,
            "image_size": [cfg.img_size, cfg.img_size],

            # ID pt evitarea data leakage-ului la ML (pacientul nu trebuie sa fie in train+test)
            "patient_session": get_patient_session(rel),

            "has_boundaries": bounds is not None,
            "boundary_csv_path": bound_path,
            "boundaries": bounds,

            "has_mask_rgb": mask_path is not None,
            "mask_rgb_path": mask_path,

            "has_bounding_boxes": len(lesions) > 0,
            "bbox_source": bbox_source,
            "num_lesions": len(lesions),
            "lesion_classes": sorted(set(l["class"] for l in lesions)),
            "total_lesion_area_percent": round(sum(l["area_percent"] for l in lesions), 2),
            "lesions": lesions,
        }

        all_meta.append(meta)

        # Salvam json individual
        jname = make_key(rel) + ".json"
        jpath = os.path.join(cfg.meta_dir, jname)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        counts["total"] += 1
        counts[disease] += 1

    # Salvam si o varianta mare (Master) care sa le aiba pe toate in acelasi fisier
    master = os.path.join(cfg.meta_dir, "_master.json")
    with open(master, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n  {'─' * 50}")
    print(f"  Total processed:      {counts['total']}")
    print(f"  With boundaries:      {counts['has_bounds']}")
    print(f"  With RGB mask:        {counts['has_mask']}")
    print(f"  With bbox (doctor):   {counts['has_bbox_doctor']}")
    print(f"  With bbox (yolo):     {counts['has_bbox_yolo']}")
    print(f"  Total cu bbox:        {counts['has_bbox_doctor'] + counts['has_bbox_yolo']}")
    print(f"\n  Per disease:")
    for d in ["AMD", "DME", "DRUSEN", "NORMAL"]:
        print(f"    {d:10s}: {counts.get(d, 0)}")

    return all_meta


def make_splits(all_meta):
    """
    v3: Split la nivel de pacient/sesiune, nu la nivel de imagine.

    Grupam imaginile dupa patient_session, facem split pe grupuri,
    apoi toate imaginile din grup merg in acelasi split.
    """
    print("  GENERATING TRAIN / VAL / TEST SPLITS (v3 - PATIENT-LEVEL)")

    # Selectam doar coloanele esentiale pt raportul final de split
    rows = [
        {
            "image_path": m["image_path"],
            "image_disk_path": m["image_disk_path"],
            "disease": m["disease_category"],
            "patient_session": m["patient_session"],
            "has_bbox": m["has_bounding_boxes"],
            "bbox_source": m["bbox_source"],
            "has_boundaries": m["has_boundaries"],
            "has_mask": m["has_mask_rgb"],
            "num_lesions": m["num_lesions"],
            "mask_rgb_path": m.get("mask_rgb_path", ""),
            "boundary_csv_path": m.get("boundary_csv_path", ""),
        }
        for m in all_meta
    ]

    df = pd.DataFrame(rows)

    # Cream DataFrame doar cu pacientii pt a face split-ul doar o data per id
    patient_df = (
        df.groupby("patient_session")
        .agg(disease=("disease", "first"))
        .reset_index()
    )

    n_patients = len(patient_df)
    print(f"\n  Total imagini:          {len(df)}")
    print(f"  Total pacienti/sesiuni: {n_patients}")
    print(f"  Medie imagini/pacient:  {len(df) / n_patients:.1f}")

    # Facem primul split (Train vs Restul) pastrand proportia claselor egala
    train_pat, temp_pat = train_test_split(
        patient_df,
        test_size=cfg.val_ratio + cfg.test_ratio,
        stratify=patient_df["disease"],
        random_state=SEED,
    )

    # Impartim 'Restul' in Validare si Testare
    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_pat, test_pat = train_test_split(
        temp_pat,
        test_size=rel_test,
        stratify=temp_pat["disease"],
        random_state=SEED,
    )

    # Castam listele in Multimi (Set) pt operatii rapide de intersectie
    train_sessions = set(train_pat["patient_session"])
    val_sessions = set(val_pat["patient_session"])
    test_sessions = set(test_pat["patient_session"])

    # Daca exista un singur pacient si in train si in val, arunca eroare instant
    assert len(train_sessions & val_sessions) == 0, "OVERLAP train/val!"
    assert len(train_sessions & test_sessions) == 0, "OVERLAP train/test!"
    assert len(val_sessions & test_sessions) == 0, "OVERLAP val/test!"
    print(f"\n  ✓ No patient overlap intre splits!")

    # Atribuim imaginile originale inapoi catre grupul stabilit per pacient
    train_df = df[df["patient_session"].isin(train_sessions)].reset_index(drop=True)
    val_df = df[df["patient_session"].isin(val_sessions)].reset_index(drop=True)
    test_df = df[df["patient_session"].isin(test_sessions)].reset_index(drop=True)

    # Salvam CSV-urile cu fisierle mapate
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(cfg.splits_dir, f"{name}.csv")
        sdf.to_csv(path, index=False)

    train_pat.to_csv(os.path.join(cfg.splits_dir, "train_patients.csv"), index=False)
    val_pat.to_csv(os.path.join(cfg.splits_dir, "val_patients.csv"), index=False)
    test_pat.to_csv(os.path.join(cfg.splits_dir, "test_patients.csv"), index=False)

    # Printeaza frumos la final tabelul
    header = f"  {'Split':<8} {'Img':>6} {'Pat':>5} {'AMD':>6} {'DME':>6} {'DRUSEN':>7} {'NORMAL':>7} {'doc':>6} {'yolo':>6} {'none':>6}"
    print(f"\n{header}")

    for name, sdf, pat_df in [
        ("train", train_df, train_pat),
        ("val", val_df, val_pat),
        ("test", test_df, test_pat),
    ]:
        dc = sdf["disease"].value_counts()
        sc = sdf["bbox_source"].value_counts()
        print(
            f"  {name:<8} {len(sdf):>6} {len(pat_df):>5} "
            f"{dc.get('AMD', 0):>6} "
            f"{dc.get('DME', 0):>6} "
            f"{dc.get('DRUSEN', 0):>7} "
            f"{dc.get('NORMAL', 0):>7} "
            f"{sc.get('doctor', 0):>6} "
            f"{sc.get('yolo', 0):>6} "
            f"{sc.get('none', 0):>6}"
        )

    print(f"\n  Splits saved to: {cfg.splits_dir}/")
    print(f"\n  IMPORTANT: splits_v3 = patient-level (no data leakage)")

    return train_df, val_df, test_df


def main():
    set_seed()

    all_meta = build_metadata()
    make_splits(all_meta)

    # Demo: Printeaza json-ul unui exemplu din consola pt a vedea exact ce structura are la final
    examples = [m for m in all_meta if m["has_bounding_boxes"] and m["has_boundaries"]]
    if examples:
        print("  EXAMPLE JSON (first image with bbox + boundaries):")
        ex = examples[0]
        show = {k: v for k, v in ex.items() if k != "boundaries"}
        show["boundaries"] = "... (see individual JSON)" if ex["boundaries"] else None
        print(json.dumps(show, indent=2, ensure_ascii=False)[:2000])

    print("  STEP 1 COMPLETE!")
    print(f"  Metadata:  {cfg.meta_dir}/")
    print(f"  Splits:    {cfg.splits_dir}/")
    print(f"  Master:    {cfg.meta_dir}/_master.json")


if __name__ == "__main__":
    main()
