"""
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
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


class Config:
    # Calea catre modelul YOLO antrenat si fisierele JSON necesare
    yolo_ckpt = "model/yoloe_oct5k.pt"
    master_json = "data/oct5k/metadata/_master.json"
    splits_dir = "data/oct5k/splits"
    out_json = "data/oct5k/yolo_bboxes.json"

    # Pragul minim de incredere (confidence) si suprapunere (IoU) pt a pastra o detectie
    conf = 0.25
    iou = 0.45


cfg = Config()

IMG_DIRS = [
    "data/OCT5k/Images/Images_Automatic",
    "data/OCT5k/Images/Images_Manual",
    "data/OCT5k/Detection/Images",
]


def locate_image(meta):
    disk = meta.get("image_disk_path", "")
    if disk and Path(disk).exists():
        return str(disk)

    # Daca locatia primara da fail, cautam relativ prin cele 3 directoare principale
    rel = meta.get("image_path", "").replace("\\", "/")
    for base in IMG_DIRS:
        full = Path(base) / rel
        if full.exists():
            return str(full)

        # Uneori in fisier scrie jpg dar fizic e png. Incercam si celelalte extensii
        for ext in [".png", ".jpeg", ".jpg"]:
            if full.with_suffix(ext).exists():
                return str(full.with_suffix(ext))
    return None


def get_doctor_image_paths(master_json):
    # Citeste master_json si colecteaza rapid path-urile imaginilor care AU DEJA marcaje manuale
    with open(master_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Folosim set comprehension pt viteza si unicitate
    return {m["image_path"] for m in metadata if m.get("has_bounding_boxes")}


def main():
    # Setam argumente CLI pt a putea suprascrie usoare setarile standard daca dorim
    parser = argparse.ArgumentParser(description="YOLOE bbox generation pe OCT5k")
    parser.add_argument("--conf", type=float, default=cfg.conf, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=cfg.iou, help="IoU threshold")
    parser.add_argument("--only-missing", action="store_true", help="Sare peste imaginile deja procesate")
    args = parser.parse_args()

    print("  YOLOE Bounding Box Generation - OCT5k")
    print(f"  Model: {cfg.yolo_ckpt}")
    print(f"  Conf threshold: {args.conf} | IoU: {args.iou}")

    # Incarcam catalogul master cu informatiile despre toate poze
    print(f"\n  Loading metadata from {cfg.master_json}...")
    with open(cfg.master_json, "r", encoding="utf-8") as f:
        master = json.load(f)

    # Le bagam intr-un dictionar indexat dupa image_path pt o cautare mult mai rapida
    metadata_dict = {m["image_path"]: m for m in master}
    print(f"  Total imagini in metadata: {len(metadata_dict)}")

    # Separam imaginile in 2 tabere: cu marcaj facut de medic vs target pt YOLO
    doctor_paths = get_doctor_image_paths(cfg.master_json)
    print(f"  Imagini cu bbox doctor: {len(doctor_paths)}")
    print(f"  Imagini fara bbox (target YOLO): {len(metadata_dict) - len(doctor_paths)}")

    # In caz ca rularea s-a oprit brutal data trecuta, putem relua lucrul incarcand json-ul parțial (--only-missing)
    existing_results = {}
    if args.only_missing and Path(cfg.out_json).exists():
        with open(cfg.out_json, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        print(f"  Deja procesate: {len(existing_results)} imagini")

    # Incarcam reteaua neuronala YOLO in memorie
    print(f"\n  Loading YOLOE from {cfg.yolo_ckpt}...")
    model = YOLO(cfg.yolo_ckpt, task="detect")
    class_names = model.names
    print(f"  YOLO classes: {class_names}")

    # Copiem rezultatele vechi pt a adauga restul procesarilor peste ele
    results = dict(existing_results)

    # Contoare pt statistici la final de script
    n_doctor = 0
    n_yolo = 0
    n_skip = 0
    n_no_img = 0
    n_no_det = 0

    all_paths = list(metadata_dict.keys())
    print(f"\n  Processing {len(all_paths)} imagini...\n")

    # Bucleaza prin absolut toate caile de imagine
    for image_path in tqdm(all_paths, desc="  YOLOE"):

        # Daca l-am calculat la o rulare trecuta, sarim peste
        if args.only_missing and image_path in existing_results:
            n_skip += 1
            continue

        meta = metadata_dict[image_path]
        is_doctor = image_path in doctor_paths

        # Daca poza e facuta de medic, vrem sa pastram informatia aia pretioasa si refuzam sa o inlocuim cu AI
        if is_doctor:
            results[image_path] = {
                "bbox_source": "doctor",
                "lesions": meta.get("lesions", []),
            }
            n_doctor += 1
            continue

        # pt restul: le cautam pe disk si le dam modelului YOLO
        img_path = locate_image(meta)

        # Daca fisierul lipseste, doar scriem eroarea si contorizam
        if img_path is None:
            results[image_path] = {
                "bbox_source": "yolo",
                "lesions": [],
                "error": "image_not_found",
            }
            n_no_img += 1
            continue

        # Rularea modelului (inferenta YOLO) - verbose=False pt a nu polua terminalul
        yolo_results = model(
            img_path,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        lesions = []

        # Daca modelul a terminat de verificat (chiar daca nu a gasit nimic, resultatul are forma specifica)
        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]

            # Daca exista cutii desenate (detectii valide)
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = class_names.get(cls_id, f"class_{cls_id}")
                    conf = round(float(box.conf.item()), 4)

                    # Luam formatul de cutie YOLO (centru_x, centru_y, latime, inaltime) in procente (normalizat 0-1)
                    xywhn = box.xywhn[0].cpu().numpy().tolist()

                    # Luam formatul de cutie Absoluta (sus_stanga_x, sus_stanga_y, jos_dr_x, jos_dr_y) in numar exact de pixeli
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = [round(v) for v in xyxy]

                    # Salvam leziunea in lista noastra
                    lesions.append({
                        "class": cls_name,
                        "bbox_yolo": [round(v, 6) for v in xywhn],
                        "bbox_abs": [x1, y1, x2, y2],
                        "confidence": conf,
                    })

        # Daca modelul n-a gasit nimic, doar marim contorul pt log
        if not lesions:
            n_no_det += 1

        # Salvam leziunile prelucrate (sau o lista goala daca e sanatos) in inregistrarea noii poze
        results[image_path] = {
            "bbox_source": "yolo",
            "lesions": lesions,
        }
        n_yolo += 1

    # Dupa ce s-a terminat bucla for, pregatim folderul pt fisierul json
    os.makedirs(os.path.dirname(cfg.out_json), exist_ok=True)

    # Scriem datele cu write mode ("w")
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculam statistica: cate dintre pozele trimise la YOLO au intors leziuni
    n_yolo_with_det = sum(
        1 for v in results.values()
        if v["bbox_source"] == "yolo" and len(v.get("lesions", [])) > 0
    )

    n_yolo_total = sum(
        1 for v in results.values()
        if v["bbox_source"] == "yolo"
    )

    print(f"  DONE!")
    print(f"  Total procesate:         {len(results)}")
    print(f"  Bbox doctor (originale): {n_doctor}")
    print(f"  Bbox YOLO generate:      {n_yolo}")
    print(f"    - Cu detectii:         {n_yolo_with_det}")
    print(f"    - Fara detectii:       {n_yolo_total - n_yolo_with_det}")
    print(f"  Imagini negasite:        {n_no_img}")
    if n_skip > 0:
        print(f"  Sarite (deja exist):   {n_skip}")
    print(f"\n  Saved: {cfg.out_json}")


if __name__ == "__main__":
    main()
