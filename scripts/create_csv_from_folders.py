import os
import pandas as pd
from pathlib import Path

# Configurare
data_root = Path("data/raw")
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]

for split in splits:
    split_path = data_root / split

    rows = []
    for class_folder in split_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            for img_path in class_folder.glob("*.*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    rows.append({
                        "image_path": str(img_path.relative_to(data_root)),
                        "label": class_name
                    })

    df = pd.DataFrame(rows)
    output_path = output_dir / f"{split}.csv"
    df.to_csv(output_path, index=False)

    print(f"{split}.csv creat: {len(df)} imagini, {df['label'].nunique()} clase")
    print(f"Distribuție: {df['label'].value_counts().to_dict()}")