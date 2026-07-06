import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# CONFIGURARE DEFAULT

DEFAULT_MODEL_PATH = "model/yolo12s_oct5k.pt"
DEFAULT_OUT_DIR = "experiments/yolo_predictions"
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# O paletă de culori frumoase și vizibile (format BGR pentru OpenCV)
COLORS = [
    (0, 255, 0),      # Verde
    (0, 165, 255),    # Portocaliu
    (0, 255, 255),    # Galben
    (255, 0, 0),      # Albastru
    (255, 0, 255),    # Magenta
    (0, 0, 255),      # Rosu
    (255, 255, 0),    # Cyan
]

def main():
    parser = argparse.ArgumentParser(description="Genereaza Bounding Boxes pe o imagine OCT folosind YOLO cu o grafica curata.")
    parser.add_argument("--image_path", type=str, required=True, help="Calea catre imaginea OCT de testat.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Calea catre modelul YOLO.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Directorul unde se va salva imaginea rezultata.")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="Confidence threshold pentru detectie.")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="IoU threshold pentru NMS.")

    args = parser.parse_args()

    # 1. Validari
    if not os.path.exists(args.image_path):
        print(f"[Eroare] Imaginea nu a fost gasita la: {args.image_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"[Eroare] Modelul YOLO nu a fost gasit la: {args.model_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  YOLO OCT BIOMARKER DETECTION (Custom Plot)")
    print(f"{'=' * 60}")
    print(f"  Imagine: {args.image_path}")
    print(f"  Model:   {args.model_path}")
    print(f"  Conf:    {args.conf} | IoU: {args.iou}")
    print(f"{'-' * 60}")

    # 2. Incarcare Model
    print("  Incarcam modelul...")
    model = YOLO(args.model_path)

    # 3. Predictie
    print("  Rulam detectia...")
    results = model.predict(
        source=args.image_path,
        conf=args.conf,
        iou=args.iou,
        verbose=False
    )

    if not results or len(results) == 0:
        print("  [Warning] Nu s-a putut procesa imaginea.")
        return

    result = results[0]
    boxes = result.boxes

    # Preluam imaginea originala pentru a desena manual pe ea
    orig_img = result.orig_img.copy()
    h, w = orig_img.shape[:2]

    # 4. Desenare Custom si Afisare
    detected_classes = []
    legend_entries = [] # Va contine tupluri: (text_legenda, culoare)

    if boxes is not None and len(boxes) > 0:
        print(f"  S-au gasit {len(boxes)} leziuni/biomarkeri:")
        for box in boxes:
            # Coordonatele box-ului
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Detalii clasa si incredere
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            conf = float(box.conf.item())

            # Alegem o culoare bazata pe ID-ul clasei
            color = COLORS[cls_id % len(COLORS)]

            detected_classes.append(cls_name)
            print(f"    - {cls_name} (Conf: {conf * 100:.1f}%)")

            # Deseneaza doar dreptunghiul, foarte curat, fara text pe imaginea propriu-zisa
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, thickness=2)

            # Adauga in lista pentru legenda
            legend_text = f"{cls_name}: {conf * 100:.1f}%"
            legend_entries.append((legend_text, color))

        # CREARE LEGENDA IN COLTUL DREAPTA-SUS
        margin = 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        line_height = 30

        # Calculam latimea necesara pentru fundalul legendei
        max_text_width = 0
        for text, _ in legend_entries:
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            if tw > max_text_width:
                max_text_width = tw

        legend_width = max_text_width + 50 # spatiu extra pentru iconita de culoare
        legend_height = len(legend_entries) * line_height + 20

        # Calculam pozitia de start (coltul dreapta-sus)
        start_x = w - legend_width - margin
        start_y = margin

        # Desenam fundalul semi-transparent (un overlay gri închis)
        overlay = orig_img.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + legend_width, start_y + legend_height), (30, 30, 30), -1)
        # Combinam overlay-ul cu imaginea (60% transparenta pentru fundalul legendei)
        cv2.addWeighted(overlay, 0.6, orig_img, 0.4, 0, orig_img)

        # Desenam textul si culorile in legenda
        current_y = start_y + 25
        for text, color in legend_entries:
            # Patratelul de culoare
            cv2.rectangle(orig_img, (start_x + 10, current_y - 12), (start_x + 25, current_y + 3), color, -1)
            # Textul cu procente
            cv2.putText(orig_img, text, (start_x + 35, current_y), font, font_scale, (255, 255, 255), text_thickness)

            current_y += line_height

    else:
        print("  Nu a fost detectat niciun biomarker (peste threshold).")

    # 5. Salvare imagine finala
    img_name = Path(args.image_path).name
    out_path = os.path.join(args.out_dir, f"pred_{img_name}")

    cv2.imwrite(out_path, orig_img)

    print(f"{'-' * 60}")
    print(f"  Imaginea cu predictii a fost salvata in:\n  -> {out_path}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()