import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OCTDataset(Dataset):

    def __init__(
        self,
        csv_path,
        data_root="data/old/raw",
        prompts_path=None,
        transform=None, # se alege ce transformrai se aplica
        tokenizer=None, # valoarea nuerica a unui cuvant(poate fi none)
        mode="train", # train/val(test)
        cache_images=False,  # tine pozele in memoria ram pt viteza sporita
    ):
        self.root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.should_cache = cache_images
        self._img_cache = {}

        self.df = pd.read_csv(csv_path) # path urile pt poze

        self.classes = sorted(self.df["label"].unique()) # sortam clasele retinei din csv
        self.label_to_int = {name: i for i, name in enumerate(self.classes)} # mapeaza clasele cu un index cnv->0

        self.prompts = None #loading prompt
        if prompts_path is not None and Path(prompts_path).exists():
            with open(prompts_path, "r") as fp:
                self.prompts = json.load(fp)
            print(f"Loaded prompts from: {prompts_path}")
        else:
            print("Image-only mode (no text prompts)")

        print(f"Dataset: {len(self.df)} images, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.df) # cate sample are datasetul (semple = imagine + label)

    def _load_image(self, path, idx):
        if self.should_cache and idx in self._img_cache:
            return self._img_cache[idx].copy()
            # returneaza copia imaginii daca este salvata in cache

        img = Image.open(path).convert("RGB")
        # deschide imaginea daca nu este salvata deja

        if self.should_cache:
            self._img_cache[idx] = img.copy()
            # salvam imagine in cache

        return img

    def _pick_prompt(self, label):
        if self.prompts is None:
            return ""

        candidates = self.prompts[label]

        if self.mode == "train":
            prompt = random.choice(candidates)
            # Mică augmentare de text: 10% șansă să adăugăm un prefix random
            if random.random() < 0.1:
                prefixes = ["An OCT scan of ", "This image shows ", "Patient with "]
                prompt = random.choice(prefixes) + prompt.lower()
            return prompt

        return candidates[0]

        candidates = self.prompts[label]
        # incarcam propturile dupa label

        if self.mode == "train":
            return random.choice(candidates)
            # alegem un propt random

        return candidates[0]

    def _tokenize(self, text):
        if self.tokenizer is None or text == "":
            return text, None

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True, # daca text e prea lung
            max_length=77,
            return_tensors="pt",
        )
        # transformam textul in token

        ids = enc["input_ids"].squeeze(0) # aplatizam tensorul de tokeni [1,77] -> [77] (1 find sizeul batchului)
        mask = enc["attention_mask"].squeeze(0) # 1 unde e text real, 0 unde e padding
        # tokeni: [234, 89, 12, 55, 7, 0, 0, 0, 0, ... 0]
        # masca: [1, 1, 1, 1, 1, 0, 0, 0, 0, ... 0]
        # filtareaza tokeni reali de cei de padding(padding e umplutura fara valoare)
        return ids, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx] # aici este indexul care are linie din csv adica o imagine + labelul
        img_file = self.root / row["image_path"]# consturim path ul catre imagine
        label_name = row["label"]
        label_int = self.label_to_int[label_name]

        img = self._load_image(img_file, idx)
        if self.transform is not None:
            img = self.transform(img)
            # aplicam transformarile setate in clasa mai sus

        prompt_text = self._pick_prompt(label_name)
        token_ids, attn_mask = self._tokenize(prompt_text)

        return {
            "image": img,
            "input_ids": token_ids,
            "attention_mask": attn_mask,
            "label": label_int,
            "label_name": label_name,
            "prompt": prompt_text,
            "image_path": str(img_file),
        }


def get_transforms(mode="train", img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Rotire mică (medicina nu suportă rotiri mari)
            transforms.RandomRotation(degrees=5),
            # Zoom și Crop (foarte important pentru a vedea detalii de edem/drusen)
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            # Doar luminozitate și contrast (fără saturație/hue pe imagini gri)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Sharpness ajută la evidențierea straturilor retinei
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            # Blur pentru a simula scanări de calitate slabă
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            normalize,
            # RandomErasing forțează modelul să nu ignore periferia imaginii
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def collate_fn_image_only(batch):
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch])
    return images, labels
    # functia de collate pt dataloader, care ia un batch de sample-uri si le combina intr-un batch de imagini si etichete
    # ex:32 dorim sa grupa 32 de imagini si labeluri [3,224,224] -> [32,3,224,224]


def collate_fn_image_text(batch):
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch]),
    }
    # functia de collate pt dataloader, care ia un batch de sample-uri si le combina intr-un batch de imagini, tokeni, masti de atentie si etichete
    # ex:32 dorim sa grupa 32 de imagini, tokeni, masti de atentie si labeluri
    # imagini: [32,3,224,224]
    # tokeni: [32,77]
    # masti de atentie: [32,77]
    # labeluri: [32]
