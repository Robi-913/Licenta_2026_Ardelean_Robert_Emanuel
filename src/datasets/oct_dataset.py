import os
import json
import random
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OCTDataset(Dataset):
    """
    Dataset PyTorch pentru OCT imagini cu prompt variation.

    Args:
        csv_path: Path către train.csv / val.csv / test.csv
        data_root: Root folder pentru imagini (ex: 'data/raw')
        prompts_path: Path către prompts.json
        transform: torchvision transforms pentru imagini
        tokenizer: Tokenizer pentru text (opțional, setat mai târziu)
        mode: 'train' sau 'eval' - controlează dacă alege prompt random
    """

    def __init__(
            self,
            csv_path,
            data_root="data/raw",
            prompts_path="data/prompts.json",
            transform=None,
            tokenizer=None,
            mode='train'
    ):
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode

        # Încarcă prompturile
        with open(prompts_path, 'r') as f:
            self.prompts = json.load(f)

        # Creează mapping label -> index pentru clasificare
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        print(f"Dataset încărcat: {len(self.df)} imagini, {len(self.classes)} clase")
        print(f"Clase: {self.classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Citește rândul din CSV
        row = self.df.iloc[idx]
        img_path = self.data_root / row['image_path']
        label = row['label']
        label_idx = self.class_to_idx[label]

        # Încarcă imaginea
        image = Image.open(img_path).convert('RGB')

        # Aplică transformări
        if self.transform:
            image = self.transform(image)

        # Alege prompt
        if self.mode == 'train':
            # Random prompt pentru training (prompt variation)
            prompt = random.choice(self.prompts[label])
        else:
            # Primul prompt pentru eval (consistency)
            prompt = self.prompts[label][0]

        # Tokenizare text (dacă avem tokenizer)
        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=77,  # CLIP standard
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            # Placeholder dacă nu avem tokenizer încă
            input_ids = prompt
            attention_mask = None

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_idx,
            'label_name': label,
            'prompt': prompt,
            'image_path': str(img_path)
        }


def get_transforms(mode='train', img_size=224):
    """
    Returnează transformări pentru imagini.

    Args:
        mode: 'train' (cu augmentare) sau 'eval' (fără augmentare)
        img_size: dimensiunea imaginii (default 224 pentru ViT)
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ============================================================================
# TEST - verificare că totul funcționează
# ============================================================================

if __name__ == "__main__":
    print("=== Test OCTDataset ===\n")

    # Creează dataset de test
    dataset = OCTDataset(
        csv_path="data/splits/train.csv",
        data_root="data/raw",
        prompts_path="data/prompts.json",
        transform=get_transforms(mode='train'),
        tokenizer=None,  # Fără tokenizer pentru test simplu
        mode='train'
    )

    print(f"\nNumăr total imagini: {len(dataset)}")
    print(f"Clase disponibile: {dataset.classes}\n")

    # Test: încarcă primul item
    sample = dataset[0]

    print("=== Sample 0 ===")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label index: {sample['label']}")
    print(f"Label name: {sample['label_name']}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Image path: {sample['image_path']}\n")

    # Test: încarcă mai multe random samples
    print("=== Random samples (prompturi diferite pentru aceeași clasă) ===")

    # Găsește toate indexurile pentru o clasă (ex: AMD)
    amd_indices = dataset.df[dataset.df['label'] == 'AMD'].index.tolist()

    for i in range(3):
        idx = random.choice(amd_indices)
        sample = dataset[idx]
        print(f"Sample {i + 1} (AMD): {sample['prompt']}")

    print("\n=== DataLoader test ===")
    from torch.utils.data import DataLoader


    # Custom collate function pentru a gestiona None
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        prompts = [item['prompt'] for item in batch]
        label_names = [item['label_name'] for item in batch]

        return {
            'image': images,
            'label': labels,
            'prompt': prompts,
            'label_name': label_names
        }


    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels: {batch['label']}")
    print(f"Batch prompts: {batch['prompt'][:2]}")  # primele 2

    print("\nEverything works!")