"""
Dataset pt MedSigLIP multi-task cu DUAL prompts + severity target

Returnează AMBELE prompturi tokenizate per imagine:
  - input_ids_a, attention_mask_a  (prompt_a = structura)
  - input_ids_b, attention_mask_b  (prompt_b = leziuni)
  - severity                       (TARGET, nu input)
  - label                          (clasa de boala)

Modelul face:
  emb_a = encode_text(prompt_a)
  emb_b = encode_text(prompt_b)
  merged = (emb_a + emb_b) / 2
  contrastive_loss(image_emb, merged)
"""

import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class OCT5kMedSigLIP(Dataset):

    def __init__(
        self,
        split_csv,
        split_json,
        severity_json,
        processor,
        image_dirs=None,
        mode="train",
    ):
        self.processor = processor
        self.mode = mode

        if image_dirs is None:
            image_dirs = [
                "data/OCT5k/Images/Images_Automatic",
                "data/OCT5k/Images/Images_Manual",
                "data/OCT5k/Detection/Images",
            ]
        self.image_dirs = image_dirs

        self.df = pd.read_csv(split_csv)

        # prompturi split
        with open(split_json, "r", encoding="utf-8") as f:
            split_list = json.load(f)

        self.split_prompts = {}
        for item in split_list:
            if item.get("split_valid") == True:
                self.split_prompts[item["image_path"]] = {
                    "a": item["prompt_a"],
                    "b": item["prompt_b"],
                }

        # severity scores
        with open(severity_json, "r", encoding="utf-8") as f:
            sev_list = json.load(f)

        self.severity = {}
        for item in sev_list:
            if item.get("severity_valid") == True and item.get("severity_percent") is not None:
                self.severity[item["image_path"]] = item["severity_percent"]

        # filtram doar ce are ambele
        valid = set(self.split_prompts.keys()) & set(self.severity.keys())
        self.df = self.df[self.df["image_path"].isin(valid)].reset_index(drop=True)

        self.classes = sorted(self.df["disease"].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        print(
            f"  OCT5kMedSigLIP [{mode}]: {len(self.df)} imagini, "
            f"{self.num_classes} clase: {self.classes}"
        )

    def __len__(self):
        return len(self.df)

    def _find_image(self, rel_path):
        normalized = rel_path.replace("\\", "/")
        for base in self.image_dirs:
            full = os.path.join(base, normalized)
            if os.path.exists(full):
                return full
            for ext in [".png", ".jpeg", ".jpg"]:
                alt = os.path.splitext(full)[0] + ext
                if os.path.exists(alt):
                    return alt
        return None

    def _tokenize_text(self, text):
        tok = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        ids = tok["input_ids"].squeeze(0)
        mask = tok.get("attention_mask", torch.ones_like(ids)).squeeze(0)
        return ids, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = self.label_to_int[row["disease"]]

        # imagine
        disk_path = self._find_image(img_path)
        if disk_path is None:
            disk_path = row.get("image_disk_path", "")
            if not os.path.exists(disk_path):
                raise FileNotFoundError(f"Nu gasesc: {img_path}")

        image = Image.open(disk_path).convert("RGB")

        # procesam imaginea
        img_inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        pixel_values = img_inputs["pixel_values"].squeeze(0)

        # tokenizam AMBELE prompturi separat
        prompts = self.split_prompts[img_path]
        ids_a, mask_a = self._tokenize_text(prompts["a"])
        ids_b, mask_b = self._tokenize_text(prompts["b"])

        # severity normalizat 0-1 (TARGET)
        severity = self.severity[img_path] / 100.0

        return {
            "pixel_values": pixel_values,
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "label": label,
            "severity": torch.tensor(severity, dtype=torch.float32),
        }


def collate_medsiglip(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids_a": torch.stack([b["input_ids_a"] for b in batch]),
        "attention_mask_a": torch.stack([b["attention_mask_a"] for b in batch]),
        "input_ids_b": torch.stack([b["input_ids_b"] for b in batch]),
        "attention_mask_b": torch.stack([b["attention_mask_b"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch]),
    }


def get_medsiglip_loaders(processor, cfg):
    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(cfg.splits_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} nu exista, skip {split}")
            continue

        ds = OCT5kMedSigLIP(
            split_csv=csv_path,
            split_json=cfg.split_json,
            severity_json=cfg.severity_json,
            processor=processor,
            mode="train" if split == "train" else "eval",
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.bs,
            shuffle=(split == "train"),
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=collate_medsiglip,
            drop_last=(split == "train"),
        )

    return loaders.get("train"), loaders.get("val"), loaders.get("test")