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


# ---------- config ----------

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

    meta_dir = "data/oct5k/metadata"
    splits_dir = "data/oct5k/splits"

    img_size = 512

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    disease_map = {
        "AMD Part1": "AMD",
        "AMD Part2": "AMD",
        "DME": "DME",
        "DRUSEN": "DRUSEN",
        "Normal Part1": "NORMAL",
        "Normal Part2": "NORMAL",
    }

    layers = ["ILM", "OPL", "IS-OS", "IBRPE", "OBRPE"]

    regions = {
        "RNFL_GCL_IPL": ("ILM", "OPL"),
        "INL_OPL": ("OPL", "IS-OS"),
        "photoreceptors": ("IS-OS", "IBRPE"),
        "RPE": ("IBRPE", "OBRPE"),
    }


cfg = Config()


# ---------- helpers ----------

def get_disease(path):
    normalized = path.replace("\\", "/")
    folder = normalized.split("/")[0]
    return cfg.disease_map.get(folder, "UNKNOWN")


def make_key(path):
    h = hashlib.md5(path.encode()).hexdigest()[:10]
    clean = path.replace("\\", "_").replace("/", "_").replace(" ", "_")
    clean = clean.replace("(", "").replace(")", "").replace(".", "_")
    clean = clean.replace("_png", "").replace("_jpeg", "").replace("_PNG", "")

    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")

    if len(clean) > 60:
        clean = clean[:60]
    return f"{clean}_{h}"


def find_image(rel_path):
    for base in cfg.img_dirs:
        full = os.path.join(base, rel_path)
        if os.path.exists(full):
            return full
        for ext in [".png", ".jpeg", ".jpg"]:
            alt = os.path.splitext(full)[0] + ext
            if os.path.exists(alt):
                return alt
    return None


def find_boundary(rel_path):
    normalized = rel_path.replace("\\", "/")
    csv_rel = os.path.splitext(normalized)[0] + ".csv"

    for base in [cfg.bounds_auto, cfg.bounds_manual]:
        full = os.path.join(base, csv_rel)
        if os.path.exists(full):
            return full
    return None


def find_mask(rel_path):
    normalized = rel_path.replace("\\", "/")
    png_rel = os.path.splitext(normalized)[0] + ".png"

    for base in [cfg.masks_auto, cfg.masks_manual]:
        full = os.path.join(base, png_rel)
        if os.path.exists(full):
            return full
    return None


def retinal_zone(cx_norm):
    if cx_norm < 0.33:
        return "nasal"
    if cx_norm < 0.66:
        return "central-foveal"
    return "temporal"


# ---------- boundary parsing ----------

def parse_boundaries(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if not all(col in df.columns for col in cfg.layers):
        return None

    layer_stats = {}
    for name in cfg.layers:
        vals = df[name].values.astype(float)
        layer_stats[name] = {
            "mean_y": round(float(np.mean(vals)), 1),
            "std_y": round(float(np.std(vals)), 2),
            "min_y": int(np.min(vals)),
            "max_y": int(np.max(vals)),
        }

    region_stats = {}
    for rname, (top, bot) in cfg.regions.items():
        top_vals = df[top].values.astype(float)
        bot_vals = df[bot].values.astype(float)
        thick = bot_vals - top_vals

        region_stats[rname] = {
            "mean_thickness_px": round(float(np.mean(thick)), 1),
            "std_thickness_px": round(float(np.std(thick)), 2),
            "min_thickness_px": int(np.min(thick)),
            "max_thickness_px": int(np.max(thick)),
            "mean_thickness_pct": round(float(np.mean(thick)) / cfg.img_size * 100, 2),
        }

    total = df["OBRPE"].values.astype(float) - df["ILM"].values.astype(float)
    t_mean = np.mean(total)
    t_std = np.std(total)

    deformations = []
    if t_std > 3:
        for x in range(len(total)):
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
        "deformation_zones": deformations[:10],
        "num_deformations": len(deformations),
    }


# ---------- bbox + layer correlation ----------

def correlate_bbox_layers(bbox, bounds):
    if bounds is None:
        return {"affected_layer": "unknown", "depth_info": "no boundary data"}

    xmin, ymin, xmax, ymax = bbox
    cy = (ymin + ymax) // 2

    layer_y = {}
    for name in cfg.layers:
        layer_y[name] = bounds["layers"][name]["mean_y"]

    ordered = sorted(layer_y.items(), key=lambda x: x[1])

    if cy < ordered[0][1]:
        return {
            "affected_layer": "above_ILM",
            "depth_info": "vitreous space, above inner limiting membrane",
            "closest_layer": ordered[0][0],
            "distance_to_closest_px": round(ordered[0][1] - cy, 1),
        }

    if cy > ordered[-1][1]:
        return {
            "affected_layer": "below_OBRPE",
            "depth_info": "choroidal space, below outer Bruch's RPE",
            "closest_layer": ordered[-1][0],
            "distance_to_closest_px": round(cy - ordered[-1][1], 1),
        }

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


def process_bboxes(img_path, bb_group, bounds):
    lesions = []

    for _, row in bb_group.iterrows():
        xmin, ymin = int(row["xmin"]), int(row["ymin"])
        xmax, ymax = int(row["xmax"]), int(row["ymax"])
        cls = row["class"]

        xmin_n = round(xmin / cfg.img_size, 4)
        ymin_n = round(ymin / cfg.img_size, 4)
        xmax_n = round(xmax / cfg.img_size, 4)
        ymax_n = round(ymax / cfg.img_size, 4)
        cx_n = (xmin_n + xmax_n) / 2
        cy_n = (ymin_n + ymax_n) / 2

        area_px = (xmax - xmin) * (ymax - ymin)
        area_pct = round(100.0 * area_px / (cfg.img_size ** 2), 2)

        layer_info = correlate_bbox_layers(
            (xmin, ymin, xmax, ymax), bounds
        )

        lesions.append({
            "class": cls,
            "bbox_px": [xmin, ymin, xmax, ymax],
            "bbox_normalized": [xmin_n, ymin_n, xmax_n, ymax_n],
            "center_normalized": [round(cx_n, 4), round(cy_n, 4)],
            "size_px": [xmax - xmin, ymax - ymin],
            "area_percent": area_pct,
            "retinal_zone": retinal_zone(cx_n),
            "layer_correlation": layer_info,
        })

    return lesions


# ---------- image collection ----------

def collect_images():
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

                if base_rel not in found or rel.endswith(".png"):
                    found[base_rel] = {
                        "rel_path": rel,
                        "disk_path": full,
                        "source": os.path.basename(base_dir),
                    }

    return found


# ---------- build metadata ----------

def build_metadata():
    print(f"{'=' * 70}")
    print("  STEP 1: BUILD STRUCTURED METADATA FOR MEDGEMMA")
    print(f"{'=' * 70}")

    os.makedirs(cfg.meta_dir, exist_ok=True)
    os.makedirs(cfg.splits_dir, exist_ok=True)

    bb_df = pd.read_csv(cfg.bb_csv)
    bb_grouped = dict(list(bb_df.groupby("image")))
    print(f"\n  Bounding boxes: {len(bb_df)} total, {len(bb_grouped)} images")

    images = collect_images()
    print(f"  Images found: {len(images)}")

    all_meta = []
    counts = defaultdict(int)

    for base_rel, info in sorted(images.items()):
        rel = info["rel_path"]
        disk = info["disk_path"]
        disease = get_disease(rel)

        bound_path = find_boundary(rel)
        bounds = None
        if bound_path:
            bounds = parse_boundaries(bound_path)
            counts["has_bounds"] += 1

        mask_path = find_mask(rel)
        if mask_path:
            counts["has_mask"] += 1

        lesions = []
        rel_norm = rel.replace("\\", "/")
        candidates = [
            rel_norm,
            rel_norm.replace(".jpeg", ".png"),
            rel_norm.replace(".jpg", ".png"),
        ]

        bb_key = None
        for c in candidates:
            if c in bb_grouped:
                bb_key = c
                break

        if bb_key:
            lesions = process_bboxes(rel, bb_grouped[bb_key], bounds)
            counts["has_bbox"] += 1

        meta = {
            "image_path": rel,
            "image_disk_path": disk,
            "disease_category": disease,
            "image_size": [cfg.img_size, cfg.img_size],

            "has_boundaries": bounds is not None,
            "boundary_csv_path": bound_path,
            "boundaries": bounds,

            "has_mask_rgb": mask_path is not None,
            "mask_rgb_path": mask_path,

            "has_bounding_boxes": len(lesions) > 0,
            "num_lesions": len(lesions),
            "lesion_classes": sorted(set(l["class"] for l in lesions)),
            "total_lesion_area_percent": round(
                sum(l["area_percent"] for l in lesions), 2
            ),
            "lesions": lesions,
        }

        all_meta.append(meta)

        jname = make_key(rel) + ".json"
        jpath = os.path.join(cfg.meta_dir, jname)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        counts["total"] += 1
        counts[disease] += 1

    master = os.path.join(cfg.meta_dir, "_master.json")
    with open(master, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n  {'─' * 50}")
    print(f"  Total processed:   {counts['total']}")
    print(f"  With boundaries:   {counts['has_bounds']}")
    print(f"  With RGB mask:     {counts['has_mask']}")
    print(f"  With bboxes:       {counts['has_bbox']}")
    print(f"\n  Per disease:")
    for d in ["AMD", "DME", "DRUSEN", "NORMAL"]:
        print(f"    {d:10s}: {counts.get(d, 0)}")

    return all_meta


# ---------- splits ----------

def make_splits(all_meta):
    print(f"\n{'=' * 70}")
    print("  GENERATING TRAIN / VAL / TEST SPLITS")
    print(f"{'=' * 70}")

    rows = []
    for m in all_meta:
        rows.append({
            "image_path": m["image_path"],
            "image_disk_path": m["image_disk_path"],
            "disease": m["disease_category"],
            "has_bbox": m["has_bounding_boxes"],
            "has_boundaries": m["has_boundaries"],
            "has_mask": m["has_mask_rgb"],
            "num_lesions": m["num_lesions"],
            "mask_rgb_path": m.get("mask_rgb_path", ""),
            "boundary_csv_path": m.get("boundary_csv_path", ""),
        })

    df = pd.DataFrame(rows)

    train_df, temp_df = train_test_split(
        df,
        test_size=cfg.val_ratio + cfg.test_ratio,
        stratify=df["disease"],
        random_state=SEED,
    )

    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["disease"],
        random_state=SEED,
    )

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(cfg.splits_dir, f"{name}.csv")
        sdf.to_csv(path, index=False)

    header = f"  {'Split':<8} {'Total':>6} {'AMD':>6} {'DME':>6} {'DRUSEN':>7} {'NORMAL':>7} {'w/bbox':>7} {'w/bound':>8} {'w/mask':>7}"
    print(f"\n{header}")
    print(f"  {'─' * 70}")

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dc = sdf["disease"].value_counts()
        print(
            f"  {name:<8} {len(sdf):>6} "
            f"{dc.get('AMD', 0):>6} "
            f"{dc.get('DME', 0):>6} "
            f"{dc.get('DRUSEN', 0):>7} "
            f"{dc.get('NORMAL', 0):>7} "
            f"{sdf['has_bbox'].sum():>7} "
            f"{sdf['has_boundaries'].sum():>8} "
            f"{sdf['has_mask'].sum():>7}"
        )

    print(f"\n  Splits saved to: {cfg.splits_dir}/")
    return train_df, val_df, test_df


# ---------- main ----------

def main():
    set_seed()

    all_meta = build_metadata()
    make_splits(all_meta)

    examples = [m for m in all_meta if m["has_bounding_boxes"] and m["has_boundaries"]]
    if examples:
        print(f"\n{'=' * 70}")
        print("  EXAMPLE JSON (first image with bbox + boundaries):")
        print(f"{'=' * 70}")
        ex = examples[0]
        show = {k: v for k, v in ex.items() if k != "boundaries"}
        show["boundaries"] = "... (see individual JSON)" if ex["boundaries"] else None
        print(json.dumps(show, indent=2, ensure_ascii=False)[:2000])

    print(f"\n{'=' * 70}")
    print("  STEP 1 COMPLETE!")
    print(f"  Metadata:  {cfg.meta_dir}/")
    print(f"  Splits:    {cfg.splits_dir}/")
    print(f"  Master:    {cfg.meta_dir}/_master.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()