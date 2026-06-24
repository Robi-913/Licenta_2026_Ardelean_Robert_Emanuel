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
            transform=None,  # functiile de augmentare/modificare poze (ex: resize, crop)
            tokenizer=None,  # modelul care transforma cuvintele in numere (ex: CLIP tokenizer)
            mode="train",  # modul in care folosim datasetul (train, val sau test)
            cache_images=False,  # daca e True, tine pozele in memoria RAM pt viteza sporita
    ):
        # salvam parametrii in interiorul clasei pentru a-i accesa mai tarziu
        self.root = Path(data_root)
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.should_cache = cache_images

        # dictionar gol in care vom salva imaginile daca cache_images e True
        self._img_cache = {}

        # citim fisierul csv cu pandas; ex: un tabel cu coloanele 'image_path' si 'label'
        self.df = pd.read_csv(csv_path)

        # extragem numele claselor unice din csv si le sortam alfabetic (ex: ['CNV', 'DME', 'DRUSEN', 'NORMAL'])
        self.classes = sorted(self.df["label"].unique())

        # cream un dictionar care asociaza un numar fiecarei clase (ex: {'CNV': 0, 'DME': 1, 'DRUSEN': 2})
        self.label_to_int = {name: i for i, name in enumerate(self.classes)}

        self.prompts = None
        # verificam daca s-a dat un path pt prompturi si daca fisierul chiar exista pe disk
        if prompts_path is not None and Path(prompts_path).exists():
            # deschidem fisierul json in modul read ('r') si il incarcam in variabila self.prompts
            with open(prompts_path, "r") as fp:
                self.prompts = json.load(fp)
            print(f"Loaded prompts from: {prompts_path}")
        else:
            print("Image-only mode (no text prompts)")

        print(f"Dataset: {len(self.df)} images, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        # returneaza cate randuri (sample-uri) are datasetul in total
        return len(self.df)

    def _load_image(self, path, idx):
        # daca cache-ul e activat si imaginea a mai fost incarcata in trecut, o luam direct din RAM
        if self.should_cache and idx in self._img_cache:
            return self._img_cache[idx].copy()

        # deschidem imaginea de pe disk si ne asiguram ca e in format RGB (3 canale de culoare)
        img = Image.open(path).convert("RGB")

        # daca cache-ul e activat, salvam o copie a imaginii incarcate in dictionar
        if self.should_cache:
            self._img_cache[idx] = img.copy()

        return img

    def _pick_prompt(self, label):
        # daca nu am incarcat niciun json cu prompturi, returnam un text gol
        if self.prompts is None:
            return ""

        # luam lista de propozitii posibile pentru clasa primita (ex: pt label='CNV' luam o lista de 5 propozitii)
        candidates = self.prompts[label]

        # la faza de antrenare (train), vrem sa adaugam diversitate
        if self.mode == "train":
            # alegem o propozitie la intamplare din lista
            prompt = random.choice(candidates)

            # mica augmentare de text: 10% sansa sa adaugam un prefix random in fata
            if random.random() < 0.1:
                prefixes = ["An OCT scan of ", "This image shows ", "Patient with "]
                prompt = random.choice(prefixes) + prompt.lower()

            return prompt

        # daca suntem in faza de testare/validare, luam mereu fix aceeasi propozitie (prima din lista) pt a fi consistenti
        return candidates[0]

    def _tokenize(self, text):
        # daca tokenizerul nu exista sau textul e gol, nu procesam nimic
        if self.tokenizer is None or text == "":
            return text, None

        # transformam propozitia in tokeni (reprezentari numerice)
        enc = self.tokenizer(
            text,
            padding="max_length",  # umplem restul spatiului cu tokeni de umplutura (padding) pana la max_length
            truncation=True,  # daca textul e prea lung, il taiem
            max_length=77,  # lungimea maxima standard ceruta de obicei de modelele tip CLIP
            return_tensors="pt",  # vrem rezultatul sub forma de tensori PyTorch ('pt')
        )

        # enc["input_ids"] are formatul [1, 77]. Aplatizam la dimensiunea [77] stergand dimensiunea 1 de batch
        ids = enc["input_ids"].squeeze(0)

        # enc["attention_mask"] ne zice ce e cuvant real (1) si ce e padding/umplutura (0)
        mask = enc["attention_mask"].squeeze(0)

        return ids, mask

    def __getitem__(self, idx):
        # luam randul corespunzator numarului 'idx' din csv
        row = self.df.iloc[idx]

        # combinam folderul principal cu calea imaginii din csv pt a obtine path-ul complet pe disk
        img_file = self.root / row["image_path"]

        # extragem numele clasei (ex: 'CNV') si numarul asociat clasei (ex: 0)
        label_name = row["label"]
        label_int = self.label_to_int[label_name]

        # incarcam efectiv poza
        img = self._load_image(img_file, idx)

        # daca avem definit un lant de transformari (ca resize, blur, etc.), il aplicam pe imagine
        if self.transform is not None:
            img = self.transform(img)

        # cautam un text potrivit pt clasa curenta si il tokenizam
        prompt_text = self._pick_prompt(label_name)
        token_ids, attn_mask = self._tokenize(prompt_text)

        # la final, functia returneaza absolut tot ce are nevoie AI-ul pt a invata din acest sample
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
    # normalizare standard folosita in modelele antrenate pe ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        # combinam mai multe transformari pt setul de antrenare
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50% sansa sa intoarca poza pe orizontala
            transforms.RandomRotation(degrees=5),  # rotire fina pt imagini medicale
            transforms.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # modificam putin lumina si contrastul
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),  # transformam imaginea in tensor matematic cu valori intre 0 si 1
            normalize,  # aplicam normalizarea de mai sus
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            # pune patrate negre pt a nu lasa modelul sa memoreze detalii false
        ])

    # la validare/testare vrem o poza curata, fara modificari haotice
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def collate_fn_image_only(batch):
    # aduna mai multe poze intr-un singur pachet de antrenare (batch)
    # ex: 32 de poze de marimea [3, 224, 224] devin un singur cub de marimea [32, 3, 224, 224]
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch])
    return images, labels


def collate_fn_image_text(batch):
    # la fel ca mai sus, dar impachetam si tokenii si mastile de atentie pentru modelul text-imagine
    return {
        "image": torch.stack([s["image"] for s in batch]),  # [batch_size, 3, 224, 224]
        "input_ids": torch.stack([s["input_ids"] for s in batch]),  # [batch_size, 77]
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),  # [batch_size, 77]
        "label": torch.tensor([s["label"] for s in batch]),  # [batch_size]
    }