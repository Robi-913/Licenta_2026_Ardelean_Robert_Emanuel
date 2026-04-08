"""
Dataset pt fine-tuning MedSigLIP contrastiv.

Încarcă imaginea OCT originală + promptul generat de MedGemma.
Folosește procesorul MedSigLIP (SigLIP) pt imagine și text.

Utilizare:
    from src.datasets.oct5k_medsiglip import OCT5kMedSigLIP, get_medsiglip_loaders
"""

import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class OCT5kMedSigLIP(Dataset):

    def __init__(
        self,
        split_csv,
        prompts_json,
        processor,
        image_dirs=None,
        mode="train",
    ):
        """
        :param split_csv: path la train.csv / val.csv / test.csv din Step 1
        :param prompts_json: path la medgemma_prompts.json din Step 2
        :param processor: SigLIP processor (AutoProcessor) pt imagine + text
        :param image_dirs: lista de foldere unde căutăm imaginile OCT
        :param mode: "train" (augmentare) sau "eval" (fără augmentare)
        """
        self.processor = processor
        self.mode = mode

        if image_dirs is None:
            image_dirs = [
                "data/OCT5k/Images/Images_Automatic",
                "data/OCT5k/Images/Images_Manual",
                "data/OCT5k/Detection/Images",
            ]
        self.image_dirs = image_dirs

        # citim split CSV
        self.df = pd.read_csv(split_csv)

        # citim prompturile MedGemma
        with open(prompts_json, "r", encoding="utf-8") as f:
            prompts_list = json.load(f)

        # dict: image_path → prompt
        self.prompts = {}
        for item in prompts_list:
            prompt = item["generated_prompt"]
            if not prompt.startswith("ERROR"):
                self.prompts[item["image_path"]] = prompt

        # filtrăm doar imaginile care au prompt valid
        self.df = self.df[self.df["image_path"].isin(self.prompts)].reset_index(drop=True)

        # clasele de boală
        self.classes = sorted(self.df["disease"].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.classes)}

        print(
            f"  OCT5kMedSigLIP [{mode}]: {len(self.df)} imagini, "
            f"{len(self.classes)} clase: {self.classes}"
        )

    def __len__(self):
        return len(self.df)

    def _find_image(self, rel_path):
        """Caută imaginea pe disk în mai multe foldere."""
        normalized = rel_path.replace("\\", "/")
        for base in self.image_dirs:
            full = os.path.join(base, normalized)
            if os.path.exists(full):
                return full
            # încearcă și alte extensii
            for ext in [".png", ".jpeg", ".jpg"]:
                alt = os.path.splitext(full)[0] + ext
                if os.path.exists(alt):
                    return alt
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        disease = row["disease"]
        label = self.label_to_int[disease]

        # încărcăm imaginea OCT originală
        disk_path = self._find_image(img_path)
        if disk_path is None:
            disk_path = row.get("image_disk_path", "")
            if not os.path.exists(disk_path):
                raise FileNotFoundError(f"Imaginea nu a fost găsită: {img_path}")

        image = Image.open(disk_path).convert("RGB")

        # promptul MedGemma
        prompt = self.prompts[img_path]

        # procesăm cu SigLIP processor (resize + normalize imagine, tokenize text)
        inputs = self.processor(
            text=prompt,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label,
            "disease": disease,
        }


def collate_medsiglip(batch):
    """Collate function pt DataLoader."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


def get_medsiglip_loaders(processor, cfg):
    """
    Creează DataLoader-ele pt train/val/test.

    :param processor: SigLIP processor (AutoProcessor)
    :param cfg: Config cu splits_dir, prompts_json, bs, workers
    :return: train_dl, val_dl, test_dl
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(cfg.splits_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} nu există, skip {split}")
            continue

        ds = OCT5kMedSigLIP(
            split_csv=csv_path,
            prompts_json=cfg.prompts_json,
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