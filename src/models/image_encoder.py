"""
Image Encoder — Vision Transformer (ViT-S/32) pentru imagini OCT.

Arhitectură:
- Input: [batch, 3, 224, 224]
- Patch embedding: împarte imaginea în patch-uri 32×32
- Transformer encoder: 6 straturi cu self-attention
- Output: [batch, embed_dim] embedding

Bazat pe ViT-S/32 din timm, dar antrenat from scratch.
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class ImageEncoder(nn.Module):
    """
    Vision Transformer pentru imagini OCT.

    Arhitectură ViT-S/32:
    - Patch size: 32×32
    - Image size: 224×224 → 7×7 = 49 patch-uri
    - Embed dim: 384 (standard pentru ViT-Small)
    - Depth: 6 straturi Transformer
    - Heads: 6 attention heads
    - MLP ratio: 4 (dimensiune hidden layer în FFN)
    """

    def __init__(
            self,
            img_size=224,  # Dimensiune imagine
            patch_size=32,  # Dimensiune patch
            embed_dim=384,  # Dimensiune embedding intern (ViT-Small standard)
            depth=6,  # Număr straturi Transformer
            num_heads=6,  # Număr attention heads
            mlp_ratio=4.0,  # Raport FFN hidden dim
            output_dim=256  # Dimensiune finală pentru contrastive learning
    ):
        """
        Inițializează Vision Transformer.

        Args:
            img_size: Dimensiune imagine (224)
            patch_size: Dimensiune patch (32)
            embed_dim: Dimensiune embedding intern (384 pentru ViT-Small)
            depth: Număr straturi Transformer (6)
            num_heads: Număr attention heads (6)
            mlp_ratio: Raport pentru MLP (4.0)
            output_dim: Dimensiune embedding final (256 pentru SigLIP)
        """
        super(ImageEncoder, self).__init__()

        # ─────────────────────────────────────────────────────────────────────
        # 1. PATCH EMBEDDING — transformă imagine în secvență de patch-uri
        # ─────────────────────────────────────────────────────────────────────
        # PatchEmbed împarte imaginea în patch-uri și le transformă în embeddings
        # Input: [batch, 3, 224, 224]
        # Output: [batch, num_patches, embed_dim] = [batch, 49, 384]
        self.patch_embed = PatchEmbed(
            img_size=img_size,  # 224
            patch_size=patch_size,  # 32
            in_chans=3,  # RGB
            embed_dim=embed_dim  # 384
        )

        num_patches = self.patch_embed.num_patches  # 49 pentru 224÷32

        # ─────────────────────────────────────────────────────────────────────
        # 2. CLS TOKEN — token special pentru agregare globală
        # ─────────────────────────────────────────────────────────────────────
        # CLS = "classification token" — se adaugă la început
        # Va fi antrenat să capteze informația globală despre imagine
        # Shape: [1, 1, embed_dim] = [1, 1, 384]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # nn.Parameter = tensor antrenabil (se actualizează în backward pass)

        # ─────────────────────────────────────────────────────────────────────
        # 3. POSITIONAL EMBEDDINGS — poziția fiecărui patch
        # ─────────────────────────────────────────────────────────────────────
        # Transformers nu au noțiune de ordine → trebuie să adăugăm poziția
        # num_patches + 1 (pentru CLS token)
        # Shape: [1, num_patches + 1, embed_dim] = [1, 50, 384]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # ─────────────────────────────────────────────────────────────────────
        # 4. DROPOUT — regularizare (previne overfitting)
        # ─────────────────────────────────────────────────────────────────────
        self.pos_drop = nn.Dropout(p=0.0)  # 0.0 = dezactivat (poți pune 0.1)

        # ─────────────────────────────────────────────────────────────────────
        # 5. TRANSFORMER BLOCKS — self-attention layers
        # ─────────────────────────────────────────────────────────────────────
        # Construim manual Transformer blocks
        # Fiecare block conține:
        # - Multi-Head Self-Attention
        # - Layer Normalization
        # - MLP (Feed-Forward Network)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=0.0,
                attn_drop=0.0
            )
            for _ in range(depth)  # 6 blocuri
        ])

        # ─────────────────────────────────────────────────────────────────────
        # 6. LAYER NORM FINAL — normalizare înainte de projection
        # ─────────────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ─────────────────────────────────────────────────────────────────────
        # 7. PROJECTION HEAD — de la embed_dim (384) la output_dim (256)
        # ─────────────────────────────────────────────────────────────────────
        # Asta e specific pentru contrastive learning
        # Proiectăm într-un spațiu mai mic unde facem dot product cu text
        self.proj = nn.Linear(embed_dim, output_dim)

        # ─────────────────────────────────────────────────────────────────────
        # INIȚIALIZARE GREUTĂȚI — important pentru convergență
        # ─────────────────────────────────────────────────────────────────────
        self._init_weights()

        print(f" ImageEncoder (ViT-S/32) creat:")
        print(f" - Patch size: {patch_size}×{patch_size}")
        print(f" - Num patches: {num_patches}")
        print(f" - Embed dim: {embed_dim}")
        print(f" - Depth: {depth} layers")
        print(f" - Heads: {num_heads}")
        print(f" - Output dim: {output_dim}")

    def _init_weights(self):
        """
        Inițializează greutățile pentru antrenare stabilă.

        Strategii:
        - CLS token: normal cu std=0.02
        - Pos embeddings: normal cu std=0.02
        - Linear layers: Xavier uniform
        - LayerNorm: bias=0, weight=1
        """
        # CLS token — mic random noise
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional embeddings — mic random noise
        nn.init.normal_(self.pos_embed, std=0.02)

        # Toate layer-urile Linear și LayerNorm
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        """
        Inițializare per-module (apelat de apply()).
        """
        if isinstance(m, nn.Linear):
            # Xavier uniform pentru weights
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass — imaginea devine embedding.

        Args:
            x: Tensor [batch_size, 3, 224, 224] — imagini RGB

        Returns:
            embeddings: Tensor [batch_size, output_dim] — image embeddings

        Flow:
            1. Patch embedding: [B, 3, 224, 224] → [B, 49, 384]
            2. Adaugă CLS token: [B, 49, 384] → [B, 50, 384]
            3. Adaugă pos embeddings: [B, 50, 384]
            4. Transformer blocks: [B, 50, 384] → [B, 50, 384]
            5. Extrage CLS: [B, 50, 384] → [B, 384]
            6. Projection: [B, 384] → [B, 256]
        """
        batch_size = x.shape[0]

        # ─────────────────────────────────────────────────────────────────────
        # 1. PATCH EMBEDDING
        # ─────────────────────────────────────────────────────────────────────
        # Transformă imagine în secvență de patch embeddings
        # [B, 3, 224, 224] → [B, 49, 384]
        x = self.patch_embed(x)

        # ─────────────────────────────────────────────────────────────────────
        # 2. ADAUGĂ CLS TOKEN
        # ─────────────────────────────────────────────────────────────────────
        # Expand CLS token pentru fiecare imagine din batch
        # self.cls_token: [1, 1, 384]
        # cls_tokens: [B, 1, 384]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenează CLS token la început
        # x: [B, 49, 384] + cls_tokens: [B, 1, 384] → [B, 50, 384]
        x = torch.cat([cls_tokens, x], dim=1)

        # ─────────────────────────────────────────────────────────────────────
        # 3. ADAUGĂ POSITIONAL EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        # Fiecare token primește informație despre poziția sa
        # x: [B, 50, 384] + pos_embed: [1, 50, 384] → [B, 50, 384]
        x = x + self.pos_embed

        # Dropout (opțional, default 0.0)
        x = self.pos_drop(x)

        # ─────────────────────────────────────────────────────────────────────
        # 4. TRANSFORMER BLOCKS — self-attention
        # ─────────────────────────────────────────────────────────────────────
        # Fiecare block aplică:
        # - Multi-head self-attention (tokens "vorbesc" între ei)
        # - Feed-forward network (transformare non-lineară)
        for block in self.blocks:
            x = block(x)  # [B, 50, 384] → [B, 50, 384]

        # ─────────────────────────────────────────────────────────────────────
        # 5. LAYER NORM FINAL
        # ─────────────────────────────────────────────────────────────────────
        x = self.norm(x)  # [B, 50, 384]

        # ─────────────────────────────────────────────────────────────────────
        # 6. EXTRAGE CLS TOKEN — primul token conține info globală
        # ─────────────────────────────────────────────────────────────────────
        # x[:, 0] = selectează primul token (CLS) din fiecare secvență
        # [B, 50, 384] → [B, 384]
        cls_output = x[:, 0]

        # ─────────────────────────────────────────────────────────────────────
        # 7. PROJECTION HEAD — proiectează în spațiul contrastive (256-dim)
        # ─────────────────────────────────────────────────────────────────────
        # [B, 384] → [B, 256]
        embeddings = self.proj(cls_output)

        return embeddings


class TransformerBlock(nn.Module):
    """
    Un singur Transformer Block.

    Conține:
    - Multi-Head Self-Attention
    - Layer Normalization
    - MLP (Feed-Forward Network)
    - Residual connections (skip connections)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        """
        Args:
            dim: Dimensiune embedding (384)
            num_heads: Număr attention heads (6)
            mlp_ratio: Raport MLP hidden dim (4.0)
            drop: Dropout rate (0.0)
            attn_drop: Attention dropout (0.0)
        """
        super(TransformerBlock, self).__init__()

        # ─────────────────────────────────────────────────────────────────────
        # LAYER NORM 1 — normalizare înainte de attention
        # ─────────────────────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(dim)

        # ─────────────────────────────────────────────────────────────────────
        # MULTI-HEAD SELF-ATTENTION
        # ─────────────────────────────────────────────────────────────────────
        # Permite fiecărui token să "vadă" toți ceilalți tokeni
        # dim = 384, num_heads = 6 → fiecare head = 384÷6 = 64 dimensiuni
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True  # Input: [batch, seq, dim] nu [seq, batch, dim]
        )

        # ─────────────────────────────────────────────────────────────────────
        # DROPOUT după attention
        # ─────────────────────────────────────────────────────────────────────
        self.drop1 = nn.Dropout(drop)

        # ─────────────────────────────────────────────────────────────────────
        # LAYER NORM 2 — normalizare înainte de MLP
        # ─────────────────────────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(dim)

        # ─────────────────────────────────────────────────────────────────────
        # MLP (FEED-FORWARD NETWORK)
        # ─────────────────────────────────────────────────────────────────────
        # dim → mlp_hidden_dim → dim
        # 384 → 1536 → 384
        mlp_hidden_dim = int(dim * mlp_ratio)  # 384 * 4 = 1536

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # Activation function (mai smooth decât ReLU)
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        Forward pass prin Transformer Block.

        Args:
            x: [batch, seq_len, dim] — ex: [32, 50, 384]

        Returns:
            x: [batch, seq_len, dim] — aceeași dimensiune

        Folosește residual connections (skip connections):
        x = x + attention(norm(x))
        x = x + mlp(norm(x))
        """
        # ─────────────────────────────────────────────────────────────────────
        # ATTENTION BLOCK cu residual connection
        # ─────────────────────────────────────────────────────────────────────
        # Normalizare → Attention → Dropout → Adună cu input original
        x_norm = self.norm1(x)

        # Self-attention: query = key = value = x_norm (de-aia e "self")
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)

        # Residual connection: x_new = x_old + attention_output
        x = x + self.drop1(attn_out)

        # ─────────────────────────────────────────────────────────────────────
        # MLP BLOCK cu residual connection
        # ─────────────────────────────────────────────────────────────────────
        # Normalizare → MLP → Adună cu input
        x = x + self.mlp(self.norm2(x))

        return x


# ═════════════════════════════════════════════════════════════════════════════
# TEST — verificare că Image Encoder funcționează
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Test ImageEncoder ===\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CREEAZĂ MODELUL
    # ─────────────────────────────────────────────────────────────────────────
    model = ImageEncoder(
        img_size=224,
        patch_size=32,
        embed_dim=384,  # ViT-Small standard
        depth=6,
        num_heads=6,
        output_dim=256  # Pentru contrastive learning
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MUTĂ PE GPU (dacă există)
    # ─────────────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TEST FORWARD PASS cu batch fake
    # ─────────────────────────────────────────────────────────────────────────
    batch_size = 8
    fake_images = torch.randn(batch_size, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(fake_images)

    print(f"Input shape:  {fake_images.shape}")  # [8, 3, 224, 224]
    print(f"Output shape: {embeddings.shape}")  # [8, 256]
    print(f"Output sample:\n{embeddings[0][:10]}...\n")  # Primele 10 valori

    # ─────────────────────────────────────────────────────────────────────────
    # 4. VERIFICARE NORMĂ L2 (opțional, pentru contrastive learning)
    # ─────────────────────────────────────────────────────────────────────────
    # În SigLIP, embeddings-urile trebuie normalizate înainte de dot product
    # Verificăm că nu sunt deja normalizate (vor fi normalizate în siglip_model.py)
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"L2 norms: {norms[:5]}")  # Ar trebui să NU fie 1.0 (nu sunt normalizate încă)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. NUMĂR PARAMETRI
    # ─────────────────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParametri totali: {total_params:,}")
    print(f"Parametri antrenabili: {trainable_params:,}")

    print("\nImageEncoder funcționează!")