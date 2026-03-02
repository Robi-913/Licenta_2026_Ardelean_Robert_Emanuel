"""
CNN Baseline — ResNet18 pentru clasificare OCT.

De ce ResNet18?
- Arhitectură dovedită (Microsoft Research, 2015)
- Nu e nici prea mic (ar underfitta) nici prea mare (ar overfitta pe 18k imagini)
- Skip connections = învață mai stabil decât CNN-uri clasice

Modificări față de ResNet18 standard:
- Ultimul layer (fc) schimbat din 1000 clase (ImageNet) → 8 clase (OCT)
- Greutăți inițializate random (from scratch, cerință profesor)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18OCT(nn.Module):
    """
    ResNet18 adaptat pentru clasificare OCT cu 8 clase.

    Moștenește din nn.Module (clasa de bază PyTorch pentru toate modelele).
    """

    def __init__(self, num_classes=8, pretrained=False):
        """
        Inițializează modelul ResNet18.

        Args:
            num_classes: Număr de clase OCT (default 8: AMD, CNV, CSR, DME, DR, DRUSEN, MH, NORMAL)
            pretrained: False = greutăți random (cerință licență from scratch)
                        True = greutăți pre-antrenate pe ImageNet (nu folosim)
        """
        # Apelează constructorul clasei părinte (PyTorch standard, obligatoriu)
        super(ResNet18OCT, self).__init__()

        # Sintaxă nouă PyTorch 0.13+ (fără warning-uri deprecation)
        # Dacă pretrained=True → folosește greutăți ImageNet
        # Dacă pretrained=False → weights=None = greutăți random
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        # Încarcă arhitectura ResNet18 din torchvision
        # ResNet18 = 18 straturi (conv layers + fc layer)
        # Are skip connections (asta îl face "Residual Network")
        self.model = models.resnet18(weights=weights)

        # Modifică ultimul layer pentru OCT
        # ResNet18 original: ultimul layer e "fc" (fully connected) cu 1000 clase (ImageNet)
        # Noi avem doar 8 clase OCT → trebuie să înlocuim fc-ul

        # Aflăm câte features intră în fc (512 pentru ResNet18)
        num_features = self.model.fc.in_features  # num_features = 512

        # Înlocuim fc-ul vechi (512 → 1000) cu fc nou (512 → 8)
        # Linear = fully connected layer
        # in_features=512 (vine din CNN), out_features=num_classes (8 clase OCT)
        self.model.fc = nn.Linear(num_features, num_classes)

        # Print pentru debug — confirmă că modelul s-a creat corect
        print(f"✓ ResNet18 creat: {num_classes} clase, weights={'pretrained' if pretrained else 'random'}")

    def forward(self, x):
        """
        Forward pass — imaginea trece prin rețea și obținem predicția.

        Asta se apelează automat când faci: output = model(image)

        Args:
            x: Tensor cu imagini [batch_size, 3, 224, 224]
               - batch_size = câte imagini procesăm deodată (ex: 32)
               - 3 = RGB (Red, Green, Blue channels)
               - 224x224 = dimensiunea imaginii (ViT standard)

        Returns:
            logits: Tensor [batch_size, num_classes]
                   - Scoruri RAW pentru fiecare clasă (înainte de softmax)
                   - Ex: [0.5, -1.2, 2.3, 0.1, -0.8, 1.5, -0.3, 0.9]
                   - Valoarea cea mai mare = clasa prezisă
        """
        # Trimite x prin întreaga rețea ResNet18 (toate cele 18 straturi)
        # self.model(x) = self.model.forward(x) implicit
        return self.model(x)


# ============================================================================
# TEST — verificare că modelul se creează și funcționează corect
# ============================================================================

if __name__ == "__main__":
    """
    Acest cod se rulează doar când faci: python src/models/cnn_resnet18.py
    NU se rulează când faci import (ex: from src.models.cnn_resnet18 import ResNet18OCT)
    """

    print("=== Test ResNet18OCT ===\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CREEAZĂ MODELUL
    # ─────────────────────────────────────────────────────────────────────────
    model = ResNet18OCT(num_classes=8, pretrained=False)
    # Creează ResNet18 cu 8 clase și greutăți random (from scratch)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MUTĂ MODELUL PE GPU (dacă există)
    # ─────────────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda' dacă ai GPU NVIDIA cu CUDA
    # device = 'cpu' dacă nu ai GPU (mult mai lent)

    model = model.to(device)
    # Mută toate parametrii modelului pe GPU (sau CPU)
    # IMPORTANT: și imaginile trebuie mutate pe același device mai târziu

    print(f"Device: {device}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CREEAZĂ UN BATCH FAKE DE IMAGINI (pentru test)
    # ─────────────────────────────────────────────────────────────────────────
    batch_size = 4  # Procesăm 4 imagini deodată
    fake_images = torch.randn(batch_size, 3, 224, 224).to(device)
    # torch.randn = numere random din distribuție normală (valori între ~-3 și ~3)
    # Shape: [4, 3, 224, 224] = 4 imagini RGB de 224x224 pixeli
    # .to(device) = mută tensorul pe GPU (trebuie să fie pe același device ca modelul)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FORWARD PASS — TRECE IMAGINILE PRIN MODEL
    # ─────────────────────────────────────────────────────────────────────────
    model.eval()
    # Pune modelul în evaluation mode
    # Asta dezactivează dropout și batch normalization (dacă ar fi)
    # IMPORTANT: în antrenare folosim model.train()

    with torch.no_grad():
        # Nu calculează gradienți (mai rapid și economisește memorie)
        # Gradienții sunt necesari doar la antrenare (backward pass)
        outputs = model(fake_images)
        # outputs = model.forward(fake_images) implicit
        # Shape: [4, 8] = 4 imagini × 8 scoruri (câte unul per clasă)

    print(f"Input shape: {fake_images.shape}")  # torch.Size([4, 3, 224, 224])
    print(f"Output shape: {outputs.shape}")  # torch.Size([4, 8])
    print(f"Output sample: {outputs[0]}\n")  # Scoruri raw pentru prima imagine

    # ─────────────────────────────────────────────────────────────────────────
    # 5. SOFTMAX — TRANSFORMĂ SCORURI ÎN PROBABILITĂȚI
    # ─────────────────────────────────────────────────────────────────────────
    probabilities = torch.softmax(outputs, dim=1)
    # dim=1 = aplică softmax pe dimensiunea claselor (8 clase)
    # Transformă scoruri raw [-inf, +inf] în probabilități [0, 1] cu suma = 1
    # Ex: [-0.3, 1.4, -3.1, ...] → [0.10, 0.53, 0.01, ...]

    print(f"Probabilități (suma = 1): {probabilities[0]}")
    print(f"Suma: {probabilities[0].sum().item():.4f}\n")  # Ar trebui să fie 1.0000

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PREDICTED CLASS — CLASA CU PROBABILITATEA CEA MAI MARE
    # ─────────────────────────────────────────────────────────────────────────
    pred_class = torch.argmax(outputs, dim=1)
    # argmax = găsește indexul valorii maxime
    # Ex: outputs[0] = [-0.3, 1.4, -3.1, ...] → argmax = 1 (clasa CNV)
    # pred_class = [1, 1, 1, 1] înseamnă că toate 4 imagini sunt clasificate ca CNV

    print(f"Predicted classes: {pred_class}")
    # Indexurile claselor: 0=AMD, 1=CNV, 2=CSR, 3=DME, 4=DR, 5=DRUSEN, 6=MH, 7=NORMAL

    # ─────────────────────────────────────────────────────────────────────────
    # 7. NUMĂR DE PARAMETRI — CÂT DE MARE E MODELUL
    # ─────────────────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    # .numel() = number of elements în tensor
    # model.parameters() = toate greutățile (weights + biases)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # requires_grad=True înseamnă că parametrul se va actualiza în antrenare
    # Pentru noi: total_params == trainable_params (totul se antrenează)

    print(f"\nParametri totali: {total_params:,}")  # ~11,180,616
    print(f"Parametri antrenabili: {trainable_params:,}")  # ~11,180,616
    # Ăștia sunt cei ~11 milioane de parametri care vor învăța din date

    print("\nModelul funcționează!")