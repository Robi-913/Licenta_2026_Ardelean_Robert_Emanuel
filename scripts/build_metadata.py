"""
Step 1: Generare JSON-uri structurate per imagine pentru pipeline-ul MedGemma

Combinăm 3 surse de date din OCT5k:
  1. Bounding boxes (Detection/all_bounding_boxes.csv) — leziuni cu coordonate
  2. Boundaries (Boundaries_Automatic/Grading/) — straturi retiniene ca CSV (ILM, OPL, IS-OS, IBRPE, OBRPE)
  3. Masks RGB (Masks_Automatic_RGB/Grading/) — segmentare vizuală a straturilor

Output:
  data/oct5k/metadata/          ← un JSON per imagine
  data/oct5k/metadata_master.json ← tot într-un fișier
  data/oct5k/splits/            ← train.csv, val.csv, test.csv

JSON-ul + masca RGB → MedGemma (Step 2) → prompt medical detaliat
Imaginea originală OCT + promptul → MedSigLIP (Step 3)

Rulare:
    python src/pipelines/step1_build_metadata.py

Notă: path-urile din Config trebuie ajustate la structura de pe mașina ta.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.seed import set_seed, SEED


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — ajustează path-urile la structura ta
# ═══════════════════════════════════════════════════════════════════════

class Config:
    # root-ul datasetului OCT5k
    oct5k_root = "data/OCT5k"

    # surse de date
    bb_csv = "data/OCT5k/Detection/all_bounding_boxes.csv"
    classes_csv = "data/OCT5k/Detection/all_classes.csv"

    # foldere imagini 512x512
    image_dirs = [
        "data/OCT5k/Images/Images_Automatic",
        "data/OCT5k/Images/Images_Manual",
        "data/OCT5k/Detection/Images",
    ]

    # boundaries (CSV cu straturi retiniene)
    boundaries_auto = "data/OCT5k/Boundaries/Boundaries_Automatic/Grading"
    # pt manual sunt 3 gradinguri, folosim doar Grading_1 ca referință
    boundaries_manual = "data/OCT5k/Boundaries/Boundaries_Manual/Grading_1"

    # măști RGB
    masks_auto_rgb = "data/OCT5k/Masks/Masks_Automatic_RGB/Grading"
    masks_manual_rgb = "data/OCT5k/Masks/Masks_Manual_RGB/Grading_1"  # sau Masks_RGB

    # output
    metadata_dir = "data/oct5k/metadata"
    splits_dir = "data/oct5k/splits"

    # imagine 512x512
    img_size = 512

    # split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # mapping disease din folder name
    disease_map = {
        "AMD Part1": "AMD",
        "AMD Part2": "AMD",
        "DME": "DME",
        "DRUSEN": "DRUSEN",
        "Normal Part1": "NORMAL",
        "Normal Part2": "NORMAL",
    }

    # straturi retiniene (din CSV boundaries)
    layer_names = ["ILM", "OPL", "IS-OS", "IBRPE", "OBRPE"]

    # regiuni retiniene între straturi
    region_names = {
        "RNFL_GCL_IPL": ("ILM", "OPL"),       # retina internă
        "INL_OPL": ("OPL", "IS-OS"),            # retina medie
        "photoreceptors": ("IS-OS", "IBRPE"),    # fotoreceptori
        "RPE": ("IBRPE", "OBRPE"),               # epiteliu pigmentar
    }


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def extract_disease(path):
    """Extrage categoria de boală din path-ul relativ al imaginii."""
    # normalizăm separatorul (Windows vs Linux)
    normalized = path.replace("\\", "/")
    first_folder = normalized.split("/")[0]
    # pt DRUSEN: path-ul e direct "DRUSEN/DRUSEN-XXXXX.png"
    return cfg.disease_map.get(first_folder, "UNKNOWN")


def safe_key(path):
    """Generează un filename flat unic din path (fără subfoldere)."""
    h = hashlib.md5(path.encode()).hexdigest()[:10]
    clean = path.replace("\\", "_").replace("/", "_").replace(" ", "_")
    clean = clean.replace("(", "").replace(")", "").replace(".", "_")
    clean = clean.replace(".png", "").replace(".jpeg", "").replace(".PNG", "")
    # eliminăm underscore-uri multiple
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if len(clean) > 60:
        clean = clean[:60]
    return f"{clean}_{h}"


def find_image(relative_path):
    """
    Caută imaginea în Images_Automatic și Images_Manual.
    Încearcă atât .png cât și .jpeg (DRUSEN).
    """
    for base_dir in cfg.image_dirs:
        full = os.path.join(base_dir, relative_path)
        if os.path.exists(full):
            return full
        # încearcă și cu extensie diferită
        for ext in [".png", ".jpeg", ".jpg"]:
            alt = os.path.splitext(full)[0] + ext
            if os.path.exists(alt):
                return alt
    return None


def find_boundary_csv(relative_path):
    """
    Caută CSV-ul cu boundaries pt o imagine.
    Path relativ: "AMD Part1/AMD (3)/Image (14).png"
    Boundary: Boundaries_Automatic/Grading/AMD Part1/AMD (3)/Image (14).csv
    DRUSEN: "DRUSEN/DRUSEN-53018-1.png" → Boundaries_Automatic/Grading/DRUSEN/DRUSEN-53018-1.csv
    """
    normalized = relative_path.replace("\\", "/")
    csv_rel = os.path.splitext(normalized)[0] + ".csv"

    for base in [cfg.boundaries_auto, cfg.boundaries_manual]:
        full = os.path.join(base, csv_rel)
        if os.path.exists(full):
            return full
    return None


def find_mask_rgb(relative_path):
    """
    Caută masca RGB pt o imagine.
    Path relativ: "AMD Part1/AMD (3)/Image (14).png"
    Mask: Masks_Automatic_RGB/Grading/AMD Part1/AMD (3)/Image (14).png
    """
    normalized = relative_path.replace("\\", "/")
    mask_rel = os.path.splitext(normalized)[0] + ".png"

    for base in [cfg.masks_auto_rgb, cfg.masks_manual_rgb]:
        full = os.path.join(base, mask_rel)
        if os.path.exists(full):
            return full
    return None


def get_retinal_zone(cx_norm):
    """
    Zona retinei pe axa x (orizontal în B-scan).
    0.0 = stânga, 1.0 = dreapta
    """
    if cx_norm < 0.33:
        return "nasal"
    elif cx_norm < 0.66:
        return "central-foveal"
    else:
        return "temporal"


# ═══════════════════════════════════════════════════════════════════════
# PARSARE BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════

def parse_boundaries(csv_path):
    """
    Parsează CSV-ul cu boundaries și extrage informații despre straturi.

    CSV format: x, ILM, OPL, IS-OS, IBRPE, OBRPE
    Fiecare rând = o coloană x din imagine (0-511)
    Valorile = coordonate y ale fiecărui strat la acel x

    Returnează dict cu:
    - grosimea medie a fiecărei regiuni
    - deviația standard (indică neregularități/deformări)
    - grosime min/max
    - total retinal thickness
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None

    if not all(col in df.columns for col in cfg.layer_names):
        return None

    layers = {}
    for name in cfg.layer_names:
        vals = df[name].values.astype(float)
        layers[name] = {
            "mean_y": round(float(np.mean(vals)), 1),
            "std_y": round(float(np.std(vals)), 2),
            "min_y": int(np.min(vals)),
            "max_y": int(np.max(vals)),
        }

    # grosimi regiuni (diferența între straturi consecutive)
    regions = {}
    for region_name, (top_layer, bottom_layer) in cfg.region_names.items():
        top_vals = df[top_layer].values.astype(float)
        bottom_vals = df[bottom_layer].values.astype(float)
        thickness = bottom_vals - top_vals  # în pixeli

        regions[region_name] = {
            "mean_thickness_px": round(float(np.mean(thickness)), 1),
            "std_thickness_px": round(float(np.std(thickness)), 2),
            "min_thickness_px": int(np.min(thickness)),
            "max_thickness_px": int(np.max(thickness)),
            "mean_thickness_pct": round(float(np.mean(thickness)) / cfg.img_size * 100, 2),
        }

    # total retinal thickness (ILM → OBRPE)
    total = df["OBRPE"].values.astype(float) - df["ILM"].values.astype(float)

    # deformări: unde grosimea deviază mult de la medie
    total_mean = np.mean(total)
    total_std = np.std(total)
    deformation_zones = []
    if total_std > 3:  # doar dacă există variație semnificativă
        for x_pos in range(len(total)):
            if abs(total[x_pos] - total_mean) > 2 * total_std:
                zone = get_retinal_zone(x_pos / cfg.img_size)
                deformation_zones.append({
                    "x_position": x_pos,
                    "x_normalized": round(x_pos / cfg.img_size, 3),
                    "zone": zone,
                    "thickness_px": int(total[x_pos]),
                    "deviation_from_mean_px": round(float(total[x_pos] - total_mean), 1),
                    "type": "thickening" if total[x_pos] > total_mean else "thinning",
                })

    return {
        "layers": layers,
        "regions": regions,
        "total_retinal_thickness": {
            "mean_px": round(float(total_mean), 1),
            "std_px": round(float(total_std), 2),
            "min_px": int(np.min(total)),
            "max_px": int(np.max(total)),
            "mean_pct": round(float(total_mean) / cfg.img_size * 100, 2),
        },
        "deformation_zones": deformation_zones[:10],  # max 10, altfel JSON-ul devine enorm
        "num_deformations": len(deformation_zones),
    }


# ═══════════════════════════════════════════════════════════════════════
# PARSARE BOUNDING BOXES — corelate cu boundaries
# ═══════════════════════════════════════════════════════════════════════

def correlate_bbox_with_layers(bbox, boundaries_data):
    """
    Determină în ce strat retinian cade centrul unui bounding box.
    Returnează stratul cel mai apropiat și poziția relativă.
    """
    if boundaries_data is None:
        return {"affected_layer": "unknown", "depth_info": "no boundary data"}

    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    # luăm valorile y ale straturilor la coloana x = cx
    layers = boundaries_data["layers"]
    layer_y_values = {}
    for name in cfg.layer_names:
        layer_y_values[name] = layers[name]["mean_y"]

    # determinăm între ce straturi cade cy
    sorted_layers = sorted(layer_y_values.items(), key=lambda x: x[1])

    if cy < sorted_layers[0][1]:
        return {
            "affected_layer": "above_ILM",
            "depth_info": "vitreous space, above inner limiting membrane",
            "closest_layer": sorted_layers[0][0],
            "distance_to_closest_px": round(sorted_layers[0][1] - cy, 1),
        }

    if cy > sorted_layers[-1][1]:
        return {
            "affected_layer": "below_OBRPE",
            "depth_info": "choroidal space, below outer Bruch's RPE",
            "closest_layer": sorted_layers[-1][0],
            "distance_to_closest_px": round(cy - sorted_layers[-1][1], 1),
        }

    # între straturi
    for i in range(len(sorted_layers) - 1):
        top_name, top_y = sorted_layers[i]
        bot_name, bot_y = sorted_layers[i + 1]
        if top_y <= cy <= bot_y:
            # mapare la regiune
            region = "unknown"
            for rname, (rtop, rbot) in cfg.region_names.items():
                if rtop == top_name and rbot == bot_name:
                    region = rname
                    break

            relative_depth = (cy - top_y) / max(1, bot_y - top_y)
            return {
                "affected_layer": region if region != "unknown" else f"between_{top_name}_and_{bot_name}",
                "between_layers": [top_name, bot_name],
                "relative_depth_in_region": round(relative_depth, 3),
                "depth_info": f"located between {top_name} (y={top_y:.0f}) and {bot_name} (y={bot_y:.0f})",
            }

    return {"affected_layer": "unknown", "depth_info": "could not determine"}


def process_bboxes_for_image(img_path, bbox_group, boundaries_data):
    """
    Procesează toate bounding boxes-urile pt o imagine.
    Returnează lista de leziuni cu metadata spațială.
    """
    lesions = []
    for _, row in bbox_group.iterrows():
        xmin, ymin = int(row["xmin"]), int(row["ymin"])
        xmax, ymax = int(row["xmax"]), int(row["ymax"])
        cls_name = row["class"]

        # coordonate normalizate
        xmin_n = round(xmin / cfg.img_size, 4)
        ymin_n = round(ymin / cfg.img_size, 4)
        xmax_n = round(xmax / cfg.img_size, 4)
        ymax_n = round(ymax / cfg.img_size, 4)
        cx_n = (xmin_n + xmax_n) / 2
        cy_n = (ymin_n + ymax_n) / 2

        # arie
        bbox_area = (xmax - xmin) * (ymax - ymin)
        area_pct = round(100.0 * bbox_area / (cfg.img_size ** 2), 2)

        # corelăm cu straturile
        layer_info = correlate_bbox_with_layers(
            (xmin, ymin, xmax, ymax), boundaries_data
        )

        lesion = {
            "class": cls_name,
            "bbox_px": [xmin, ymin, xmax, ymax],
            "bbox_normalized": [xmin_n, ymin_n, xmax_n, ymax_n],
            "center_normalized": [round(cx_n, 4), round(cy_n, 4)],
            "size_px": [xmax - xmin, ymax - ymin],
            "area_percent": area_pct,
            "retinal_zone": get_retinal_zone(cx_n),
            "layer_correlation": layer_info,
        }
        lesions.append(lesion)

    return lesions


# ═══════════════════════════════════════════════════════════════════════
# COLECTARE TOATE IMAGINILE
# ═══════════════════════════════════════════════════════════════════════

def collect_all_images():
    """
    Colectăm toate imaginile din Images_Automatic și Images_Manual.
    Returnăm un dict cu path relativ → path pe disk.
    Eliminăm duplicatele (aceeași imagine poate fi în ambele foldere).
    """
    all_images = {}

    for base_dir in cfg.image_dirs:
        if not os.path.exists(base_dir):
            print(f"  WARNING: {base_dir} nu există!")
            continue

        for root, _, files in os.walk(base_dir):
            for fname in files:
                if not fname.lower().endswith((".png", ".jpeg", ".jpg")):
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, base_dir)

                # preferăm varianta .png peste .jpeg
                base_rel = os.path.splitext(rel_path)[0]
                if base_rel not in all_images or rel_path.endswith(".png"):
                    all_images[base_rel] = {
                        "rel_path": rel_path,
                        "disk_path": full_path,
                        "source": os.path.basename(base_dir),
                    }

    return all_images


# ═══════════════════════════════════════════════════════════════════════
# MAIN: BUILD METADATA
# ═══════════════════════════════════════════════════════════════════════

def build_all_metadata():
    """
    Construiește JSON-ul structurat pt fiecare imagine din OCT5k.
    """
    print(f"{'=' * 70}")
    print("  STEP 1: BUILD STRUCTURED METADATA FOR MEDGEMMA")
    print(f"{'=' * 70}")

    os.makedirs(cfg.metadata_dir, exist_ok=True)
    os.makedirs(cfg.splits_dir, exist_ok=True)

    # 1. Citim bounding boxes
    bb_df = pd.read_csv(cfg.bb_csv)
    bb_grouped = dict(list(bb_df.groupby("image")))
    print(f"\n  Bounding boxes: {len(bb_df)} total, {len(bb_grouped)} imagini")

    # 2. Colectăm toate imaginile
    all_images = collect_all_images()
    print(f"  Imagini găsite: {len(all_images)}")

    # 3. Procesăm fiecare imagine
    all_metadata = []
    stats = defaultdict(int)

    for base_rel, img_info in sorted(all_images.items()):
        rel_path = img_info["rel_path"]
        disk_path = img_info["disk_path"]
        disease = extract_disease(rel_path)

        # căutăm boundary CSV
        boundary_path = find_boundary_csv(rel_path)
        boundaries_data = None
        if boundary_path:
            boundaries_data = parse_boundaries(boundary_path)
            stats["has_boundaries"] += 1

        # căutăm masca RGB
        mask_path = find_mask_rgb(rel_path)
        if mask_path:
            stats["has_mask"] += 1

        # căutăm bounding boxes
        # normalizăm path-urile la forward slash pt matching (BB csv folosește /)
        lesions = []
        bb_key = None
        rel_normalized = rel_path.replace("\\", "/")
        candidates = [
            rel_normalized,
            rel_normalized.replace(".jpeg", ".png"),
            rel_normalized.replace(".jpg", ".png"),
        ]
        for candidate in candidates:
            if candidate in bb_grouped:
                bb_key = candidate
                break

        if bb_key:
            lesions = process_bboxes_for_image(rel_path, bb_grouped[bb_key], boundaries_data)
            stats["has_bboxes"] += 1

        # construim metadata
        meta = {
            "image_path": rel_path,
            "image_disk_path": disk_path,
            "disease_category": disease,
            "image_size": [cfg.img_size, cfg.img_size],

            # boundaries
            "has_boundaries": boundaries_data is not None,
            "boundary_csv_path": boundary_path,
            "boundaries": boundaries_data,

            # masca RGB
            "has_mask_rgb": mask_path is not None,
            "mask_rgb_path": mask_path,

            # bounding boxes / leziuni
            "has_bounding_boxes": len(lesions) > 0,
            "num_lesions": len(lesions),
            "lesion_classes": sorted(set(l["class"] for l in lesions)),
            "total_lesion_area_percent": round(
                sum(l["area_percent"] for l in lesions), 2
            ),
            "lesions": lesions,
        }

        all_metadata.append(meta)

        # salvăm JSON individual
        json_name = safe_key(rel_path) + ".json"
        json_path = os.path.join(cfg.metadata_dir, json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        stats["total"] += 1
        stats[disease] += 1

    # salvăm master JSON
    master_path = os.path.join(cfg.metadata_dir, "_master.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    # raport
    print(f"\n  {'─' * 50}")
    print(f"  Total imagini procesate: {stats['total']}")
    print(f"  Cu boundaries:          {stats['has_boundaries']}")
    print(f"  Cu mască RGB:           {stats['has_mask']}")
    print(f"  Cu bounding boxes:      {stats['has_bboxes']}")
    print(f"\n  Per boală:")
    for d in ["AMD", "DME", "DRUSEN", "NORMAL"]:
        print(f"    {d:10s}: {stats.get(d, 0)}")

    return all_metadata


# ═══════════════════════════════════════════════════════════════════════
# SPLITS
# ═══════════════════════════════════════════════════════════════════════

def generate_splits(all_metadata):
    """
    Generează train/val/test splits stratificate pe disease_category.
    """
    print(f"\n{'=' * 70}")
    print("  GENERATING TRAIN / VAL / TEST SPLITS")
    print(f"{'=' * 70}")

    records = []
    for meta in all_metadata:
        records.append({
            "image_path": meta["image_path"],
            "image_disk_path": meta["image_disk_path"],
            "disease": meta["disease_category"],
            "has_bbox": meta["has_bounding_boxes"],
            "has_boundaries": meta["has_boundaries"],
            "has_mask": meta["has_mask_rgb"],
            "num_lesions": meta["num_lesions"],
            "mask_rgb_path": meta.get("mask_rgb_path", ""),
            "boundary_csv_path": meta.get("boundary_csv_path", ""),
        })

    df = pd.DataFrame(records)

    # stratified split
    train_df, temp_df = train_test_split(
        df, test_size=(cfg.val_ratio + cfg.test_ratio),
        stratify=df["disease"], random_state=SEED,
    )
    relative_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["disease"], random_state=SEED,
    )

    # salvăm
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(cfg.splits_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)

    # raport
    print(f"\n  {'Split':<8} {'Total':>6} {'AMD':>6} {'DME':>6} {'DRUSEN':>7} {'NORMAL':>7} {'w/bbox':>7} {'w/bound':>8} {'w/mask':>7}")
    print(f"  {'─' * 70}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dc = split_df["disease"].value_counts()
        print(
            f"  {name:<8} {len(split_df):>6} "
            f"{dc.get('AMD', 0):>6} "
            f"{dc.get('DME', 0):>6} "
            f"{dc.get('DRUSEN', 0):>7} "
            f"{dc.get('NORMAL', 0):>7} "
            f"{split_df['has_bbox'].sum():>7} "
            f"{split_df['has_boundaries'].sum():>8} "
            f"{split_df['has_mask'].sum():>7}"
        )

    print(f"\n  Splits salvate în: {cfg.splits_dir}/")
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    set_seed()

    all_metadata = build_all_metadata()
    generate_splits(all_metadata)

    # exemplu: afișăm un JSON pt o imagine cu bounding boxes
    examples = [m for m in all_metadata if m["has_bounding_boxes"] and m["has_boundaries"]]
    if examples:
        print(f"\n{'=' * 70}")
        print("  EXEMPLU JSON (prima imagine cu bbox + boundaries):")
        print(f"{'=' * 70}")
        ex = examples[0]
        # afișăm fără boundaries complete (prea lung)
        display = {k: v for k, v in ex.items() if k != "boundaries"}
        display["boundaries"] = "... (vezi JSON individual)" if ex["boundaries"] else None
        print(json.dumps(display, indent=2, ensure_ascii=False)[:2000])

    print(f"\n{'=' * 70}")
    print("  STEP 1 COMPLETE!")
    print(f"  Metadata:  {cfg.metadata_dir}/")
    print(f"  Splits:    {cfg.splits_dir}/")
    print(f"  Master:    {cfg.metadata_dir}/_master.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()