import pandas as pd
from pathlib import Path


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

data_root = Path("data/raw") # datasetul
out_dir = Path("data/splits") # unde se salveaza
out_dir.mkdir(parents=True, exist_ok=True)


def scan_folder(folder):
    entries = []
    for cls_dir in sorted(folder.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.glob("*.*"):
            if img.suffix.lower() not in IMG_EXTENSIONS:
                continue
            entries.append({
                "image_path": str(img.relative_to(data_root)),
                "label": cls_dir.name,
            })
    return entries
    # cautam alfabetic in director si mapam fiecare imagine gasita cu numele folderului parinte (eticheta),
    # salvand calea relativa in lista

for split in ["train", "val", "test"]:
    rows = scan_folder(data_root / split)
    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{split}.csv"
    df.to_csv(csv_path, index=False)

    counts = df["label"].value_counts().to_dict()
    print(f"{split}.csv: {len(df)} images, {df['label'].nunique()} classes")
    print(f"Distribution: {counts}")
    # impartim datele in train, val, test, generam csvurile aferente si verificam volumul de imagini si distributia claselor