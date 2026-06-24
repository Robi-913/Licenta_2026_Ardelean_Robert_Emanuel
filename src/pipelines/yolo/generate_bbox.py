"""
YOLOE Bounding Box Generation pe toate imaginile OCT5k

Ruleaza YOLOE pe toate 4573 imaginile si salveaza bbox-urile detectate.
Imaginile cu bbox de la doctori (566) sunt marcate ca "doctor".
Restul (4007) sunt marcate ca "yolo".

Output: data/oct5k/yolo_bboxes.json
Format:
{
  "image_path": {
    "bbox_source": "doctor" | "yolo",
    "lesions": [
      {
        "class": "Softdrusen",
        "bbox_yolo": [x_center, y_center, width, height],  # normalized 0-1
        "bbox_abs": [x1, y1, x2, y2],                     # pixeli absoluti
        "confidence": 0.87
      }
    ]
  }
}

Rulare:
    python src/pipeline/yolo/generate_bbox.py
    python src/pipeline/yolo/generate_bbox.py --conf 0.25  # threshold default
    python src/pipeline/yolo/generate_bbox.py --only-missing  # skip deja procesate
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# ================================================================
# CONFIG
# ================================================================

class Config:
    yolo_ckpt   = "models/yoloe_oct5k.pt"
    master_json = "data/oct5k/metadata/_master.json"
    splits_dir  = "data/oct5k/splits"
    out_json    = "data/oct5k/yolo_bboxes.json"

    conf        = 0.25   # confidence threshold
    iou         = 0.45   # IoU threshold

cfg = Config()

IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


# ================================================================
# HELPERS
# ================================================================

def locate_image(meta):
    """Gaseste imaginea OCT originala pe disk."""
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


def get_doctor_image_paths(master_json):
    """Returneaza set-ul de image_path-uri care au bbox de la doctori."""
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    doctor_paths = set()
    for m in metadata:
        if m.get("has_bounding_boxes"):
            doctor_paths.add(m["image_path"])
    return doctor_paths


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOE bbox generation pe OCT5k")
    parser.add_argument("--conf", type=float, default=cfg.conf, help="Confidence threshold")
    parser.add_argument("--iou",  type=float, default=cfg.iou,  help="IoU threshold")
    parser.add_argument("--only-missing", action="store_true",
                        help="Sare peste imaginile deja procesate")
    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOE Bounding Box Generation — OCT5k")
    print(f"  Model: {cfg.yolo_ckpt}")
    print(f"  Conf threshold: {args.conf} | IoU: {args.iou}")
    print("=" * 70)

    # incarca master metadata
    print(f"\n  Loading metadata from {cfg.master_json}...")
    with open(cfg.master_json, "r", encoding="utf-8") as f:
        master = json.load(f)
    metadata_dict = {m["image_path"]: m for m in master}
    print(f"  Total imagini in metadata: {len(metadata_dict)}")

    # imaginile cu bbox de la doctori
    doctor_paths = get_doctor_image_paths(cfg.master_json)
    print(f"  Imagini cu bbox doctor: {len(doctor_paths)}")
    print(f"  Imagini fara bbox (target YOLO): {len(metadata_dict) - len(doctor_paths)}")

    # incarca rezultate existente daca --only-missing
    existing_results = {}
    if args.only_missing and Path(cfg.out_json).exists():
        with open(cfg.out_json, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        print(f"  Deja procesate: {len(existing_results)} imagini")

    # incarca YOLO
    print(f"\n  Loading YOLOE from {cfg.yolo_ckpt}...")
    model = YOLO(cfg.yolo_ckpt, task="detect")
    class_names = model.names
    print(f"  YOLO classes: {class_names}")

    # rezultate finale
    results = dict(existing_results)
    n_doctor  = 0
    n_yolo    = 0
    n_skip    = 0
    n_no_img  = 0
    n_no_det  = 0

    all_paths = list(metadata_dict.keys())
    print(f"\n  Processing {len(all_paths)} imagini...\n")

    for image_path in tqdm(all_paths, desc="  YOLOE"):

        # skip daca deja procesat
        if args.only_missing and image_path in existing_results:
            n_skip += 1
            continue

        meta     = metadata_dict[image_path]
        is_doctor = image_path in doctor_paths

        # pentru imaginile cu bbox doctor — pastram bbox-urile originale
        if is_doctor:
            results[image_path] = {
                "bbox_source": "doctor",
                "lesions": meta.get("lesions", []),
            }
            n_doctor += 1
            continue

        # pentru restul — ruleaza YOLOE
        img_path = locate_image(meta)
        if img_path is None:
            results[image_path] = {
                "bbox_source": "yolo",
                "lesions": [],
                "error": "image_not_found",
            }
            n_no_img += 1
            continue

        # YOLO inference
        yolo_results = model(
            img_path,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        lesions = []
        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]
            img_h, img_w = r.orig_shape

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id   = int(box.cls.item())
                    cls_name = class_names.get(cls_id, f"class_{cls_id}")
                    conf     = round(float(box.conf.item()), 4)

                    # bbox normalized (YOLO format: x_center, y_center, w, h)
                    xywhn = box.xywhn[0].cpu().numpy().tolist()

                    # bbox absolut (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = [round(v) for v in xyxy]

                    lesions.append({
                        "class":      cls_name,
                        "bbox_yolo":  [round(v, 6) for v in xywhn],
                        "bbox_abs":   [x1, y1, x2, y2],
                        "confidence": conf,
                    })

        if not lesions:
            n_no_det += 1

        results[image_path] = {
            "bbox_source": "yolo",
            "lesions":     lesions,
        }
        n_yolo += 1

    # salveaza
    os.makedirs(os.path.dirname(cfg.out_json), exist_ok=True)
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # statistici finale
    n_yolo_with_det = sum(
        1 for v in results.values()
        if v["bbox_source"] == "yolo" and len(v.get("lesions", [])) > 0
    )
    n_yolo_total = sum(
        1 for v in results.values()
        if v["bbox_source"] == "yolo"
    )

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"{'=' * 70}")
    print(f"  Total procesate:        {len(results)}")
    print(f"  Bbox doctor (originale): {n_doctor}")
    print(f"  Bbox YOLO generate:     {n_yolo}")
    print(f"    - Cu detectii:        {n_yolo_with_det}")
    print(f"    - Fara detectii:      {n_yolo_total - n_yolo_with_det}")
    print(f"  Imagini negasite:       {n_no_img}")
    if n_skip > 0:
        print(f"  Sarite (deja exist):  {n_skip}")
    print(f"\n  Saved: {cfg.out_json}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()