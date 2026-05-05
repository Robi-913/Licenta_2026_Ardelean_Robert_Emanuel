import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset


class OCT5kDataset(Dataset):

    def __init__(self, split_csv, split_json, severity_json, processor,
                 img_dirs=None, mode="train"):
        self.processor = processor
        self.mode = mode

        if img_dirs is None:
            img_dirs = [
                "data/OCT5k/Images/Images_Automatic",
                "data/OCT5k/Images/Images_Manual",
                "data/OCT5k/Detection/Images",
            ]
        self.img_dirs = img_dirs

        self.df = pd.read_csv(split_csv)

        with open(split_json, "r", encoding="utf-8") as f:
            raw_splits = json.load(f)

        self.prompts = {}
        for entry in raw_splits:
            if entry.get("split_valid") is True:
                self.prompts[entry["image_path"]] = {
                    "a": entry["prompt_a"],
                    "b": entry["prompt_b"],
                }

        with open(severity_json, "r", encoding="utf-8") as f:
            raw_sev = json.load(f)

        self.sev = {}
        for entry in raw_sev:
            pct = entry.get("severity_percent")
            if entry.get("severity_valid") is True and pct is not None:
                self.sev[entry["image_path"]] = pct

        usable = set(self.prompts.keys()) & set(self.sev.keys())
        self.df = self.df[self.df["image_path"].isin(usable)].reset_index(drop=True)

        self.classes = sorted(self.df["disease"].unique())
        self.lbl_map = {name: i for i, name in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        print(
            f"  OCT5k [{mode}]: {len(self.df)} images, "
            f"{self.n_classes} classes: {self.classes}"
        )

    def __len__(self):
        return len(self.df)

    def _locate(self, rel):
        norm = rel.replace("\\", "/")
        for base in self.img_dirs:
            full = os.path.join(base, norm)
            if os.path.exists(full):
                return full
            for ext in [".png", ".jpeg", ".jpg"]:
                alt = os.path.splitext(full)[0] + ext
                if os.path.exists(alt):
                    return alt
        return None

    def _auto_crop(self, img, threshold=35):
        """Taie marginile negre automat, pastreaza zona cu continut retinian."""
        arr = np.array(img.convert("L"))
        mask = arr > threshold

        rows = mask.any(axis=1)
        cols = mask.any(axis=0)

        if rows.any() and cols.any():
            y1 = int(rows.argmax())
            y2 = int(len(rows) - rows[::-1].argmax())
            x1 = int(cols.argmax())
            x2 = int(len(cols) - cols[::-1].argmax())

            pad = 5
            y1 = max(0, y1 - pad)
            x1 = max(0, x1 - pad)
            y2 = min(arr.shape[0], y2 + pad)
            x2 = min(arr.shape[1], x2 + pad)

            if (x2 - x1) > 50 and (y2 - y1) > 50:
                img = img.crop((x1, y1, x2, y2))

        return img

    def _tok(self, text):
        enc = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc.get("attention_mask", torch.ones_like(ids)).squeeze(0)
        return ids, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = self.lbl_map[row["disease"]]

        disk = self._locate(img_path)
        if disk is None:
            disk = row.get("image_disk_path", "")
            if not os.path.exists(disk):
                raise FileNotFoundError(f"Cannot find: {img_path}")

        img = Image.open(disk).convert("RGB")

        # 1. denoise INTAI — reduce speckle noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # 2. auto-crop DUPA — masca detecteaza mai curat fara noise
        img = self._auto_crop(img)

        # 3. flip doar la train
        if self.mode == "train":
            img = T.RandomHorizontalFlip(p=0.5)(img)

        px = self.processor(images=img, return_tensors="pt")
        pixels = px["pixel_values"].squeeze(0)

        pair = self.prompts[img_path]
        ids_a, mask_a = self._tok(pair["a"])
        ids_b, mask_b = self._tok(pair["b"])

        sev = self.sev[img_path] / 100.0

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
    out = {}

    for split in ["train", "val", "test"]:
        csv = os.path.join(cfg.splits_dir, f"{split}.csv")
        if not os.path.exists(csv):
            print(f"  WARNING: {csv} missing, skipping {split}")
            continue

        ds = OCT5kDataset(
            split_csv=csv,
            split_json=cfg.split_json,
            severity_json=cfg.severity_json,
            processor=processor,
            mode="train" if split == "train" else "eval",
        )

        is_train = split == "train"
        out[split] = DataLoader(
            ds,
            batch_size=cfg.bs,
            shuffle=is_train,
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=collate_oct5k,
            drop_last=is_train,
        )

    return out.get("train"), out.get("val"), out.get("test")