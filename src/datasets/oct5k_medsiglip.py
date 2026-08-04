import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset


class OCT5kDataset(Dataset):

    def __init__(
            self,
            split_csv,
            split_json,
            severity_json,
            processor,
            img_dirs=None,
            mode="train"
    ):
        self.processor = processor
        self.mode = mode

        self.img_dirs = img_dirs or [
            "data/OCT5k/Images/Images_Automatic",
            "data/OCT5k/Images/Images_Manual",
            "data/OCT5k/Detection/Images",
        ]

        # Incarcam tabelul cu informatiile despre imagini
        self.df = pd.read_csv(split_csv)

        # 1. Incarcam JSON-ul cu prompturi (textele descriptive)
        with open(split_json, "r", encoding="utf-8") as f:
            raw_splits = json.load(f)

        self.prompts = {}
        # Verificam formatul fisierului JSON ca sa suportam atat versiuni vechi cat si noi
        if isinstance(raw_splits, dict):
            # Structura este tip dictionar (ex: {"poza.jpg": {"a": "text", "b": "text"}})
            for path, entry in raw_splits.items():
                a = entry.get("a", "")
                b = entry.get("b", "")
                if a and b:
                    self.prompts[path] = {"a": a, "b": b}

        # 2. Incarcam JSON-ul cu scorurile de severitate ale bolii
        with open(severity_json, "r", encoding="utf-8") as f:
            raw_sev = json.load(f)

        self.sev = {}
        for entry in raw_sev:
            pct = entry.get("severity_percent")
            # Adaugam in dictionar doar intrarile marcate ca valide si care au efectiv un procentaj
            if entry.get("severity_valid") is True and pct is not None:
                path = entry["image_path"].replace("/", "\\")
                self.sev[path] = pct

        # 3. Sincronizam formatul path-urilor peste tot
        self.prompts = {k.replace("/", "\\"): v for k, v in self.prompts.items()}
        self.df["image_path"] = self.df["image_path"].apply(lambda p: p.replace("/", "\\"))

        # 4. Filtram setul de date: pastram DOAR imaginile care au si prompt valid si severitate valida
        usable = set(self.prompts.keys()) & set(self.sev.keys())
        self.df = self.df[self.df["image_path"].isin(usable)].reset_index(drop=True)

        # Extragem si sortam clasele unice de boala (ex: ['DME', 'NORMAL'])
        self.classes = sorted(self.df["disease"].unique())
        self.lbl_map = {name: i for i, name in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        print(
            f"OCT5k [{mode}]: {len(self.df)} images, "
            f"{self.n_classes} classes: {self.classes}"
        )

    def __len__(self):
        # Numarul total de sample-uri procesabile
        return len(self.df)

    def _locate(self, rel):
        # Functie care cauta o imagine in lista de foldere posibile (img_dirs)
        norm = rel.replace("\\", "/")
        for base in self.img_dirs:
            full = os.path.join(base, norm)
            if os.path.exists(full):
                return full

            # Uneori extensia din csv difera de cea reala (ex: .png vs .jpg), asa ca incercam mai multe variante
            for ext in [".png", ".jpeg", ".jpg"]:
                alt = os.path.splitext(full)[0] + ext
                if os.path.exists(alt):
                    return alt
        return None

    def _auto_crop(self, img, threshold=35):
        # Taie marginile negre automat si pastreaza doar zona centrala utila (retina)
        # Convertim imaginea in alb-negru ('L') pentru a putea izola usor lumina de intuneric
        arr = np.array(img.convert("L"))

        # Cream o masca logica (True unde e mai luminos decat 35, False pt negru/fundal)
        mask = arr > threshold

        # Verificam care randuri si coloane au macar un pixel luminos (deci fac parte din ochi)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)

        # Daca poza nu e complet neagra (are randuri/coloane valide)
        if rows.any() and cols.any():
            # Gasim indexul primei si ultimei linii/coloane cu continut
            y1, y2 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
            x1, x2 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())

            # Adaugam un padding (margine) de 5 pixeli de siguranta, fara sa iesim din dimensiunile imaginii
            pad = 5
            y1 = max(0, y1 - pad)
            x1 = max(0, x1 - pad)
            y2 = min(arr.shape[0], y2 + pad)
            x2 = min(arr.shape[1], x2 + pad)

            # Evitam sa decupam ceva gresit daca zona ramasa ar fi extrem de mica (ex: doar zgomot)
            if (x2 - x1) > 50 and (y2 - y1) > 50:
                img = img.crop((x1, y1, x2, y2))

        return img

    def _tok(self, text):
        # Transforma textul brut in indecsi numerici recunoscuti de model
        enc = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        # Scoaterea dimensiunii extra de batch
        ids = enc["input_ids"].squeeze(0)
        # Asigurare ca masca de atentie e generata (unele tokenizere vechi cer fallback)
        mask = enc.get("attention_mask", torch.ones_like(ids)).squeeze(0)

        return ids, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = self.lbl_map[row["disease"]]

        # Cautam imaginea pe disc
        disk = self._locate(img_path)
        if disk is None:
            disk = row.get("image_disk_path", "")
            if not os.path.exists(disk):
                raise FileNotFoundError(f"Cannot find image: {img_path}")

        img = Image.open(disk).convert("RGB")

        # 1. Denoise INTAI: aplicam un blur subtil pentru a elimina noise-ul fin (speckle noise)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # 2. Auto-crop DUPA denoise: masca detecteaza mult mai curat muchiile fara punctulete de noise
        img = self._auto_crop(img)

        # 3. Augmentare: doar cand antrenam facem flip orizontal (pt generalizare)
        if self.mode == "train":
            img = T.RandomHorizontalFlip(p=0.5)(img)

        # Procesam imaginea pt model (echivalentul a normalizare + ToTensor pt modele specifice gen CLIP)
        px = self.processor(images=img, return_tensors="pt")
        pixels = px["pixel_values"].squeeze(0)

        # Preluam prompturile corespunzatoare pozei si le tokenizam pe ambele
        pair = self.prompts[img_path]
        ids_a, mask_a = self._tok(pair["a"])
        ids_b, mask_b = self._tok(pair["b"])

        # Transformam severitatea din procentaj (ex: 85%) in valoare subunitara (0.85)
        sev = self.sev[img_path] / 100.0

        # Returnam un pachet complet cu imagini, texte multiple si severitate
        return {
            "pixel_values": pixels,
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "label": label,
            "severity": torch.tensor(sev, dtype=torch.float32),
        }


def collate_oct5k(batch):
    # Functia care uneste individualitatile in grupuri (batch-uri) pt trecerea prin placa video
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids_a": torch.stack([b["input_ids_a"] for b in batch]),
        "attention_mask_a": torch.stack([b["attention_mask_a"] for b in batch]),
        "input_ids_b": torch.stack([b["input_ids_b"] for b in batch]),
        "attention_mask_b": torch.stack([b["attention_mask_b"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch]),
    }


def make_loaders(processor, cfg):
    # Dictionar in care vom tine loaderele pt antrenare, validare si testare
    out = {}

    for split in ["train", "val", "test"]:
        csv = os.path.join(cfg.splits_dir, f"{split}.csv")

        # Daca un anumit fisier csv lipseste, il saram (ex. poate nu vrem validare curenta)
        if not os.path.exists(csv):
            print(f"WARNING: {csv} missing, skipping {split}")
            continue

        # Generam Obiectul Dataset specific etapei curente (train/val/test)
        ds = OCT5kDataset(
            split_csv=csv,
            split_json=cfg.split_json,
            severity_json=cfg.severity_json,
            processor=processor,
            mode="train" if split == "train" else "eval",
        )

        # Setam optiunile DataLoader-ului (doar setul de train trebuie sa fie amestecat 'shuffle=True')
        is_train = split == "train"
        out[split] = DataLoader(
            ds,
            batch_size=cfg.bs,  # Ex: 32 imagini per grup
            shuffle=is_train,  # Amesteca la fiecare epocha doar la train
            num_workers=cfg.workers,  # Numarul de thread-uri de procesor pt incarcat fisiere de pe disc
            pin_memory=True,  # Muta mai repede din RAM pe GPU
            collate_fn=collate_oct5k,  # Functia definita mai sus pt ambalare
            drop_last=is_train,  # Daca ultimul batch are ex. doar 5 imagini in loc de 32, e ignorat la train
        )

    # Returnam loaderele impachetate
    return out.get("train"), out.get("val"), out.get("test")
