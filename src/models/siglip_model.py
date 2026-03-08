"""
SigLIP Model — wrapper complet peste Image Encoder + Text Encoder.

Combină cele 2 modele și normalizează embeddings-urile pentru contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# Adaugă root-ul proiectului la sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder


class SigLIPModel(nn.Module):
    """
    Model multimodal SigLIP pentru contrastive learning imagine-text.

    Componente:
    - Image Encoder (ViT-S/32)
    - Text Encoder (Transformer)
    - L2 Normalization pe ambele embeddings

    La antrenare:
    - Primește batch de imagini + texte pereche
    - Returnează embeddings normalizate
    - Loss-ul (SigLIP) se calculează extern

    La inferență (zero-shot):
    - encode_image(): imagine → embedding
    - encode_text(): text → embedding
    - Similaritate = dot product între embeddings normalizate
    """

    def __init__(
            self,
            # Image encoder params
            img_size=224,
            patch_size=32,
            image_embed_dim=384,
            image_depth=6,
            image_num_heads=6,

            # Text encoder params
            vocab_size=30522,
            max_seq_len=77,
            text_embed_dim=256,
            text_depth=4,
            text_num_heads=4,
            text_pooling='mean',

            # Shared params
            output_dim=256  # Dimensiune comună pentru ambele embeddings
    ):
        """
        Inițializează SigLIP Model.

        Args:
            Image encoder params: vezi ImageEncoder
            Text encoder params: vezi TextEncoder
            output_dim: Dimensiune finală pentru embeddings (256)
        """
        super(SigLIPModel, self).__init__()

        self.output_dim = output_dim

        # ─────────────────────────────────────────────────────────────────────
        # 1. IMAGE ENCODER — ViT pentru imagini OCT
        # ─────────────────────────────────────────────────────────────────────
        self.image_encoder = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=image_embed_dim,
            depth=image_depth,
            num_heads=image_num_heads,
            output_dim=output_dim
        )

        # ─────────────────────────────────────────────────────────────────────
        # 2. TEXT ENCODER — Transformer pentru prompturi
        # ─────────────────────────────────────────────────────────────────────
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=text_embed_dim,
            depth=text_depth,
            num_heads=text_num_heads,
            output_dim=output_dim,
            pooling=text_pooling
        )

        # ─────────────────────────────────────────────────────────────────────
        # 3. TEMPERATURE — parametru antrenabil pentru scaling similaritate
        # ─────────────────────────────────────────────────────────────────────
        # În SigLIP/CLIP, temperatura scalează dot product-ul
        # Valoare inițială: log(1/0.07) ≈ 2.66 (standard CLIP)
        # Antrenabil: se va ajusta automat în training
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))

        print(f"\n SigLIP Model creat:")
        print(f" Image Encoder: {sum(p.numel() for p in self.image_encoder.parameters()):,} params")
        print(f" Text Encoder:  {sum(p.numel() for p in self.text_encoder.parameters()):,} params")
        print(f" Output dim: {output_dim}")
        print(f" Total params: {sum(p.numel() for p in self.parameters()):,}")

    def encode_image(self, images):
        """
        Encoder imagini → embeddings normalizate.

        Args:
            images: Tensor [batch_size, 3, 224, 224]

        Returns:
            image_embeddings: Tensor [batch_size, output_dim], norm=1.0
        """
        # Image encoder: [B, 3, 224, 224] → [B, 256]
        embeddings = self.image_encoder(images)

        # L2 normalization: embeddings / ||embeddings||_2
        # Normă L2 = sqrt(sum(x^2))
        # După normalizare: ||embeddings||_2 = 1.0
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode_text(self, input_ids, attention_mask=None):
        """
        Encoder text → embeddings normalizate.

        Args:
            input_ids: Tensor [batch_size, seq_len] — token IDs
            attention_mask: Tensor [batch_size, seq_len] — 1=real, 0=padding

        Returns:
            text_embeddings: Tensor [batch_size, output_dim], norm=1.0
        """
        # Text encoder: [B, L] → [B, 256]
        embeddings = self.text_encoder(input_ids, attention_mask)

        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, images, input_ids, attention_mask=None):
        """
        Forward pass complet — pentru antrenare.

        Args:
            images: Tensor [batch_size, 3, 224, 224]
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len]

        Returns:
            image_embeddings: Tensor [batch_size, output_dim], normalizat
            text_embeddings: Tensor [batch_size, output_dim], normalizat
            logit_scale: Scalar tensor — temperatura pentru scaling

        Usage în training:
            image_emb, text_emb, logit_scale = model(images, input_ids, attn_mask)
            loss = siglip_loss(image_emb, text_emb, logit_scale)
        """
        # Encode imagini
        image_embeddings = self.encode_image(images)  # [B, 256], norm=1.0

        # Encode text
        text_embeddings = self.encode_text(input_ids, attention_mask)  # [B, 256], norm=1.0

        # Returnează embeddings + temperatura
        return image_embeddings, text_embeddings, self.logit_scale


# ═════════════════════════════════════════════════════════════════════════════
# TEST — verificare că SigLIP Model funcționează
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Test SigLIPModel ===\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CREEAZĂ MODELUL
    # ─────────────────────────────────────────────────────────────────────────
    model = SigLIPModel(
        img_size=224,
        patch_size=32,
        image_embed_dim=384,
        image_depth=6,
        image_num_heads=6,

        vocab_size=30522,
        max_seq_len=77,
        text_embed_dim=256,
        text_depth=4,
        text_num_heads=4,
        text_pooling='mean',

        output_dim=256
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MUTĂ PE GPU
    # ─────────────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TEST FORWARD PASS cu batch fake
    # ─────────────────────────────────────────────────────────────────────────
    batch_size = 8

    # Fake images
    fake_images = torch.randn(batch_size, 3, 224, 224).to(device)

    # Fake text (token IDs + attention mask)
    seq_len = 20
    fake_input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    fake_attention_mask = torch.ones(batch_size, seq_len).to(device)

    model.eval()
    with torch.no_grad():
        image_emb, text_emb, logit_scale = model(
            fake_images,
            fake_input_ids,
            fake_attention_mask
        )

    print(f"Images shape:       {fake_images.shape}")  # [8, 3, 224, 224]
    print(f"Input IDs shape:    {fake_input_ids.shape}")  # [8, 20]
    print(f"\nImage embeddings:   {image_emb.shape}")  # [8, 256]
    print(f"Text embeddings:    {text_emb.shape}")  # [8, 256]
    print(f"Logit scale:        {logit_scale.item():.4f}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. VERIFICARE NORMALIZARE L2 — trebuie să fie 1.0!
    # ─────────────────────────────────────────────────────────────────────────
    image_norms = torch.norm(image_emb, p=2, dim=1)
    text_norms = torch.norm(text_emb, p=2, dim=1)

    print(f"Image L2 norms: {image_norms[:5]}")  # Ar trebui să fie ~1.0
    print(f"Text L2 norms:  {text_norms[:5]}\n")  # Ar trebui să fie ~1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TEST SIMILARITATE — dot product între embeddings normalizate
    # ─────────────────────────────────────────────────────────────────────────
    # Matricea de similaritate B×B
    # similarity[i, j] = dot(image_emb[i], text_emb[j])
    # Diagonal = perechi pozitive (imagine i cu text i)
    similarity = image_emb @ text_emb.T  # [B, B]

    print(f"Similarity matrix shape: {similarity.shape}")  # [8, 8]
    print(f"Similarity matrix:\n{similarity}\n")

    # Diagonal ar trebui să aibă valori mai mari (perechi pozitive)
    # Off-diagonal = perechi negative (ar trebui mai mici)
    diagonal = torch.diag(similarity)
    print(f"Diagonal (pozitiv): {diagonal}")
    print(f"Mean diagonal: {diagonal.mean():.4f}")
    print(f"Mean off-diagonal: {(similarity.sum() - diagonal.sum()) / (batch_size * (batch_size - 1)):.4f}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. NUMĂR PARAMETRI
    # ─────────────────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri antrenabili: {trainable_params:,}")

    print("\nSigLIPModel funcționează!")