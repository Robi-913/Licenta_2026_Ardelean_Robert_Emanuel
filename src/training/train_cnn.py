"""
Script de antrenare pentru CNN Baseline (ResNet18).

Flow complet:
1. Încarcă datele (train + validation)
2. Creează modelul ResNet18
3. Definește loss function (CrossEntropyLoss) și optimizer (Adam)
4. Loop antrenare: pentru fiecare epoch
   - Forward pass: imaginea → predicție
   - Calculează loss (eroarea)
   - Backward pass: calculează gradienți
   - Update greutăți
   - Evaluare pe validation
5. Salvează modelul și metricile
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

# Adaugă root-ul proiectului la sys.path ca să poți face import din src/
# Asta permite: from src.datasets.oct_dataset import OCTDataset
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.oct_dataset import OCTDataset, get_transforms
from src.models.cnn_resnet18 import ResNet18OCT
from src.utils.seed import set_seed


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Antrenează modelul pentru un epoch (o trecere prin tot dataset-ul de train).

    Args:
        model: ResNet18OCT
        train_loader: DataLoader cu imagini de train
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: 'cuda' sau 'cpu'
        epoch: Numărul epoch-ului curent (pentru logging)

    Returns:
        avg_loss: Loss-ul mediu pe tot epoch-ul
        accuracy: Accuracy mediu pe train
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Pune modelul în TRAINING MODE
    # ─────────────────────────────────────────────────────────────────────────
    model.train()
    # Activează dropout (dacă ar fi) și batch normalization în mod train
    # Opusul: model.eval() — folosit la evaluare

    running_loss = 0.0  # Acumulăm loss-ul pentru toate batch-urile
    all_preds = []  # Lista cu toate predicțiile
    all_labels = []  # Lista cu toate label-urile adevărate

    # ─────────────────────────────────────────────────────────────────────────
    # LOOP PRIN TOATE BATCH-URILE
    # ─────────────────────────────────────────────────────────────────────────
    # tqdm = progress bar fancy (arată progresul în terminal)
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in train_bar:
        # ─────────────────────────────────────────────────────────────────────
        # 1. EXTRAGE DATELE DIN BATCH
        # ─────────────────────────────────────────────────────────────────────
        images = batch['image'].to(device)  # Shape: [batch_size, 3, 224, 224]
        labels = batch['label'].to(device)  # Shape: [batch_size]
        # .to(device) = mută datele pe GPU (trebuie să fie pe același device ca modelul)

        # ─────────────────────────────────────────────────────────────────────
        # 2. ZERO GRADIENTS — resetează gradienții din iterația anterioară
        # ─────────────────────────────────────────────────────────────────────
        optimizer.zero_grad()
        # De ce? Gradienții se acumulează în PyTorch (sum by default)
        # Dacă nu faci zero_grad(), gradienții noi se ADUNĂ la cei vechi → bug

        # ─────────────────────────────────────────────────────────────────────
        # 3. FORWARD PASS — trece imaginile prin model
        # ─────────────────────────────────────────────────────────────────────
        outputs = model(images)  # Shape: [batch_size, num_classes]
        # outputs = scoruri raw (logits) pentru fiecare clasă
        # Ex: [[-0.3, 1.4, -3.1, 0.2, ...], [...], ...]

        # ─────────────────────────────────────────────────────────────────────
        # 4. CALCULEAZĂ LOSS (eroarea)
        # ─────────────────────────────────────────────────────────────────────
        loss = criterion(outputs, labels)
        # CrossEntropyLoss face două lucruri:
        # 1. Aplică softmax pe outputs (transformă în probabilități)
        # 2. Calculează -log(probabilitate_clasa_corecta)
        # Loss mare = predicție proastă; Loss mic = predicție bună

        # ─────────────────────────────────────────────────────────────────────
        # 5. BACKWARD PASS — calculează gradienții
        # ─────────────────────────────────────────────────────────────────────
        loss.backward()
        # Calculează ∂loss/∂weight pentru FIECARE parametru din model
        # Asta e backpropagation — algoritmul care permite învățarea
        # Gradienții spun: "în ce direcție să modifici weight-ul ca să scazi loss-ul"

        # ─────────────────────────────────────────────────────────────────────
        # 6. UPDATE WEIGHTS — actualizează parametrii modelului
        # ─────────────────────────────────────────────────────────────────────
        optimizer.step()
        # Aplică regula: weight_nou = weight_vechi - learning_rate * gradient
        # Ex: dacă gradient = -0.5 și lr = 0.001 → weight crește cu 0.0005
        # Asta e momentul în care modelul ÎNVAȚĂ

        # ─────────────────────────────────────────────────────────────────────
        # 7. TRACKING — salvează loss și predicțiile pentru metrici
        # ─────────────────────────────────────────────────────────────────────
        running_loss += loss.item()  # .item() = extrage valoarea numerică din tensor

        # Predicted class = indexul cu scorul maxim
        preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size]

        all_preds.extend(preds.cpu().numpy())  # Mută pe CPU și transformă în listă
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar cu loss-ul curent
        train_bar.set_postfix({'loss': loss.item()})

    # ─────────────────────────────────────────────────────────────────────────
    # CALCULEAZĂ METRICI PENTRU ÎNTREG EPOCH-UL
    # ─────────────────────────────────────────────────────────────────────────
    avg_loss = running_loss / len(train_loader)  # Loss mediu per batch
    accuracy = accuracy_score(all_labels, all_preds)  # % predicții corecte

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluează modelul pe validation set (fără antrenare).

    Args:
        model: ResNet18OCT
        val_loader: DataLoader cu imagini de validation
        criterion: Loss function
        device: 'cuda' sau 'cpu'

    Returns:
        avg_loss: Loss mediu pe validation
        accuracy: Accuracy pe validation
        f1: F1 score macro (media F1-urilor per clasă)
        all_preds: Lista cu toate predicțiile (pentru confusion matrix)
        all_labels: Lista cu toate label-urile adevărate
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Pune modelul în EVALUATION MODE
    # ─────────────────────────────────────────────────────────────────────────
    model.eval()
    # Dezactivează dropout și pune batch normalization în eval mode
    # IMPORTANT: la evaluare nu vrem randomness

    running_loss = 0.0
    all_preds = []
    all_labels = []

    # ─────────────────────────────────────────────────────────────────────────
    # NO GRADIENTS — nu calculăm gradienți la evaluare
    # ─────────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        # Asta economisește memorie și accelerează evaluarea
        # Nu avem nevoie de gradienți pentru că NU facem backward pass

        val_bar = tqdm(val_loader, desc="Validation")

        for batch in val_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass (fără backward)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            val_bar.set_postfix({'loss': loss.item()})

    # ─────────────────────────────────────────────────────────────────────────
    # CALCULEAZĂ METRICI
    # ─────────────────────────────────────────────────────────────────────────
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # F1 MACRO = media F1-urilor pentru fiecare clasă
    # De ce macro? Pentru că tratează fiecare clasă egal (important când e balanced dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1, all_preds, all_labels


def plot_training_history(history, save_path):
    """
    Generează grafice cu loss și accuracy pe parcursul antrenării.

    Args:
        history: Dict cu {train_loss: [...], val_loss: [...], val_acc: [...], val_f1: [...]}
        save_path: Path unde să salveze figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFIC 1: LOSS (train vs validation)
    # ─────────────────────────────────────────────────────────────────────────
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFIC 2: ACCURACY + F1 pe validation
    # ─────────────────────────────────────────────────────────────────────────
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o', color='green')
    axes[1].plot(history['val_f1'], label='Val F1 (macro)', marker='s', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Grafic salvat: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Generează confusion matrix.

    Args:
        y_true: Label-uri adevărate
        y_pred: Label-uri prezise
        class_names: Lista cu numele claselor ['AMD', 'CNV', ...]
        save_path: Path unde să salveze figura
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix salvată: {save_path}")
    plt.close()


def train_cnn():
    """
    Funcția principală de antrenare — orchestrează tot flow-ul.
    """
    print("=" * 70)
    print("ANTRENARE CNN BASELINE (ResNet18)")
    print("=" * 70)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. CONFIGURARE
    # ═════════════════════════════════════════════════════════════════════════

    # Hyperparametri
    NUM_EPOCHS = 30  # De câte ori trece prin tot dataset-ul
    BATCH_SIZE = 32  # Câte imagini procesează deodată
    LEARNING_RATE = 0.001  # Cât de mari sunt pașii de învățare (1e-3)
    NUM_WORKERS = 0  # Câte procese paralele pentru încărcare date
    NUM_CLASSES = 4  # Numărul de clase folosit la clasificare

    # Paths
    OUTPUT_DIR = Path("experiments/cnn_baseline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Reproducibility — setează seed pentru rezultate consistente
    set_seed(42)  # 42 = seed standard (orice număr merge)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. ÎNCARCĂ DATELE
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[1/6] Încărcare date...")

    # Custom collate function (fix pentru None în attention_mask)
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'image': images, 'label': labels}

    # Dataset train
    train_dataset = OCTDataset(
        csv_path="data/old/splits/train.csv",
        data_root="data/old/raw",
        prompts_path="data/old/prompts.json",
        transform=get_transforms(mode='train'),  # Cu augmentare
        tokenizer=None,  # Nu avem nevoie de tokenizer pentru CNN
        mode='train'
    )

    # Dataset validation
    val_dataset = OCTDataset(
        csv_path="data/old/splits/val.csv",
        data_root="data/old/raw",
        prompts_path="data/old/prompts.json",
        transform=get_transforms(mode='eval'),  # Fără augmentare
        tokenizer=None,
        mode='eval'
    )

    # DataLoaders — facilitează iterarea prin date în batch-uri
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Amestecă datele la fiecare epoch (important!)
        num_workers=NUM_WORKERS,  # Procese paralele pentru încărcare
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Nu amestecăm la validation
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)} imagini, {len(train_loader)} batch-uri")
    print(f"Val: {len(val_dataset)} imagini, {len(val_loader)} batch-uri")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. CREEAZĂ MODELUL
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[2/6] Creare model...")

    model = ResNet18OCT(num_classes=NUM_CLASSES, use_pretrained=False)
    model = model.to(device)  # Mută modelul pe GPU

    # ═════════════════════════════════════════════════════════════════════════
    # 4. LOSS FUNCTION & OPTIMIZER
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[3/6] Configurare loss & optimizer...")

    # CrossEntropyLoss = loss function standard pentru clasificare multi-clasă
    # Combină softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer = algoritm de optimizare popular (mai bun decât SGD pentru majoritatea cazurilor)
    # lr = learning rate (cât de mari sunt pașii de update)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Loss: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. TRAINING LOOP — AICI SE ÎNTÂMPLĂ ÎNVĂȚAREA
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n[4/6] Antrenare ({NUM_EPOCHS} epoci)...")

    # Dict pentru salvare istoric antrenare
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    best_val_f1 = 0.0  # Tracking pentru best model

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 70}")

        # ─────────────────────────────────────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # ─────────────────────────────────────────────────────────────────────
        # PRINT RESULTS
        # ─────────────────────────────────────────────────────────────────────
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # ─────────────────────────────────────────────────────────────────────
        # SALVARE ISTORIC
        # ─────────────────────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # ─────────────────────────────────────────────────────────────────────
        # SALVARE BEST MODEL (bazat pe F1 score)
        # ─────────────────────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = CHECKPOINT_DIR / "best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Greutățile modelului
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"  Best model salvat (F1: {val_f1:.4f})")

    print("\n" + "=" * 70)
    print(f"ANTRENARE COMPLETĂ | Best Val F1: {best_val_f1:.4f}")
    print("=" * 70)

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SALVARE REZULTATE FINALE
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Salvare rezultate...")

    # Salvează istoric ca CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"Metrici salvate: {OUTPUT_DIR / 'metrics.csv'}")

    # Generează grafice
    plot_training_history(history, OUTPUT_DIR / "loss_curves.png")

    # Confusion matrix (pe ultimul epoch)
    class_names = train_dataset.classes
    plot_confusion_matrix(val_labels, val_preds, class_names, OUTPUT_DIR / "confusion_matrix.png")

    # Copiază best model în folder principal checkpoints
    best_checkpoint_src = CHECKPOINT_DIR / "best.pth"
    best_checkpoint_dst = Path("checkpoints/resnet18_final.pth")
    best_checkpoint_dst.parent.mkdir(exist_ok=True)

    import shutil
    shutil.copy(best_checkpoint_src, best_checkpoint_dst)
    print(f"Best model copiat în: {best_checkpoint_dst}")

    print("\n[6/6] GATA! CNN Baseline antrenat cu succes.")
    print(f"\nRezultate finale:")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {best_checkpoint_dst}")
    print(f"  Metrici: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"  Grafice: {OUTPUT_DIR / 'loss_curves.png'}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_cnn()