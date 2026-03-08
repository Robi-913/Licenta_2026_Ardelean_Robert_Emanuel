"""
Text Encoder — Transformer pentru procesare text prompturi OCT.

Arhitectură:
- Input: token IDs [batch, seq_len]
- Word embeddings + Positional embeddings
- Transformer encoder: 4-6 straturi cu self-attention
- Mean pooling sau CLS token pentru agregare
- Output: [batch, output_dim] embedding

Antrenat from scratch pentru contrastive learning cu imagini OCT.
"""

import torch
import torch.nn as nn
import math


class TextEncoder(nn.Module):
    """
    Transformer Encoder pentru text prompturi.

    Arhitectură:
    - Vocab size: 30522 (BERT tokenizer standard)
    - Max sequence length: 77 (standard CLIP)
    - Embed dim: 256 (mai mic decât BERT pentru eficiență)
    - Depth: 4 straturi (suficient pentru prompturi scurte)
    - Heads: 4 attention heads
    - Output dim: 256 (pentru contrastive learning)
    """

    def __init__(
            self,
            vocab_size=30522,  # BERT tokenizer vocab
            max_seq_len=77,  # CLIP standard
            embed_dim=256,  # Dimensiune internă
            depth=4,  # Număr straturi Transformer
            num_heads=4,  # Număr attention heads
            mlp_ratio=4.0,  # Raport MLP hidden dim
            output_dim=256,  # Dimensiune finală
            pooling='mean'  # 'mean' sau 'cls'
    ):
        """
        Inițializează Text Encoder.

        Args:
            vocab_size: Dimensiune vocabular (30522 pentru BERT)
            max_seq_len: Lungime maximă secvență (77)
            embed_dim: Dimensiune embedding intern (256)
            depth: Număr straturi Transformer (4)
            num_heads: Număr attention heads (4)
            mlp_ratio: Raport MLP (4.0)
            output_dim: Dimensiune embedding final (256)
            pooling: Tip agregare — 'mean' (media pe tokens) sau 'cls' (primul token)
        """
        super(TextEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pooling = pooling

        # ─────────────────────────────────────────────────────────────────────
        # 1. WORD EMBEDDINGS — fiecare token ID → vector embed_dim
        # ─────────────────────────────────────────────────────────────────────
        # Tabel de embeddings: vocab_size × embed_dim
        # Ex: token_id=2019 ("OCT") → vector [256]
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # ─────────────────────────────────────────────────────────────────────
        # 2. POSITIONAL EMBEDDINGS — poziția fiecărui token în secvență
        # ─────────────────────────────────────────────────────────────────────
        # Max length × embed_dim
        # Token la poziția 0 primește pos_embed[0], poziția 1 primește pos_embed[1], etc.
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # ─────────────────────────────────────────────────────────────────────
        # 3. DROPOUT — regularizare
        # ─────────────────────────────────────────────────────────────────────
        self.embed_drop = nn.Dropout(p=0.1)  # 10% dropout

        # ─────────────────────────────────────────────────────────────────────
        # 4. TRANSFORMER BLOCKS — self-attention layers
        # ─────────────────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=0.1,
                attn_drop=0.0
            )
            for _ in range(depth)  # 4 blocuri
        ])

        # ─────────────────────────────────────────────────────────────────────
        # 5. LAYER NORM FINAL
        # ─────────────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ─────────────────────────────────────────────────────────────────────
        # 6. PROJECTION HEAD — de la embed_dim (256) la output_dim (256)
        # ─────────────────────────────────────────────────────────────────────
        # În cazul nostru embed_dim == output_dim, dar le păstrăm separate
        # pentru flexibilitate
        if embed_dim != output_dim:
            self.proj = nn.Linear(embed_dim, output_dim)
        else:
            self.proj = nn.Identity()  # Nu face nimic dacă dimensiunile sunt egale

        # ─────────────────────────────────────────────────────────────────────
        # INIȚIALIZARE GREUTĂȚI
        # ─────────────────────────────────────────────────────────────────────
        self._init_weights()

        print(f" TextEncoder creat:")
        print(f" - Vocab size: {vocab_size}")
        print(f" - Max seq len: {max_seq_len}")
        print(f" - Embed dim: {embed_dim}")
        print(f" - Depth: {depth} layers")
        print(f" - Heads: {num_heads}")
        print(f" - Pooling: {pooling}")
        print(f" - Output dim: {output_dim}")

    def _init_weights(self):
        """Inițializează greutățile."""
        # Token embeddings — normal distribution
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Positional embeddings — normal distribution
        nn.init.normal_(self.positional_embedding.weight, std=0.02)

        # Aplică inițializare pentru toate layer-urile
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        """Inițializare per-module."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass — text devine embedding.

        Args:
            input_ids: Tensor [batch_size, seq_len] — token IDs
            attention_mask: Tensor [batch_size, seq_len] — 1 pentru tokeni reali, 0 pentru padding

        Returns:
            embeddings: Tensor [batch_size, output_dim] — text embeddings

        Flow:
            1. Token embeddings: [B, L] → [B, L, embed_dim]
            2. Positional embeddings: [B, L, embed_dim]
            3. Sum: token_emb + pos_emb
            4. Transformer blocks: [B, L, embed_dim] → [B, L, embed_dim]
            5. Pooling (mean sau cls): [B, L, embed_dim] → [B, embed_dim]
            6. Projection: [B, embed_dim] → [B, output_dim]
        """
        batch_size, seq_len = input_ids.shape

        # ─────────────────────────────────────────────────────────────────────
        # 1. TOKEN EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        # Fiecare token ID → vector embedding
        # input_ids: [B, L] → token_emb: [B, L, embed_dim]
        token_emb = self.token_embedding(input_ids)

        # ─────────────────────────────────────────────────────────────────────
        # 2. POSITIONAL EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        # Creează indici de poziție: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, L]
        positions = positions.expand(batch_size, -1)  # [B, L]

        # Lookup positional embeddings
        # positions: [B, L] → pos_emb: [B, L, embed_dim]
        pos_emb = self.positional_embedding(positions)

        # ─────────────────────────────────────────────────────────────────────
        # 3. COMBINĂ TOKEN + POSITIONAL EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        # Element-wise addition
        x = token_emb + pos_emb  # [B, L, embed_dim]

        # Dropout
        x = self.embed_drop(x)

        # ─────────────────────────────────────────────────────────────────────
        # 4. TRANSFORMER BLOCKS — self-attention
        # ─────────────────────────────────────────────────────────────────────
        # Convertește attention_mask pentru PyTorch Transformer
        # PyTorch expects: True = ignore, False = attend
        # Noi avem: 1 = attend, 0 = ignore → inversăm
        if attention_mask is not None:
            # [B, L] → [B, L] cu True/False
            # 0 (padding) → True (ignore), 1 (real token) → False (attend)
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Aplică Transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)  # [B, L, embed_dim]

        # ─────────────────────────────────────────────────────────────────────
        # 5. LAYER NORM
        # ─────────────────────────────────────────────────────────────────────
        x = self.norm(x)  # [B, L, embed_dim]

        # ─────────────────────────────────────────────────────────────────────
        # 6. POOLING — agregare de la [B, L, embed_dim] la [B, embed_dim]
        # ─────────────────────────────────────────────────────────────────────
        if self.pooling == 'cls':
            # Extrage primul token (asemănător cu BERT [CLS])
            pooled = x[:, 0, :]  # [B, embed_dim]

        elif self.pooling == 'mean':
            # Mean pooling peste tokeni (ignoră padding)
            if attention_mask is not None:
                # Calculează media doar peste tokeni reali (unde mask=1)
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
                sum_embeddings = (x * mask_expanded).sum(dim=1)  # [B, embed_dim]
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1], evită div by 0
                pooled = sum_embeddings / sum_mask  # [B, embed_dim]
            else:
                # Mean peste toți tokenii
                pooled = x.mean(dim=1)  # [B, embed_dim]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # ─────────────────────────────────────────────────────────────────────
        # 7. PROJECTION HEAD
        # ─────────────────────────────────────────────────────────────────────
        embeddings = self.proj(pooled)  # [B, embed_dim] → [B, output_dim]

        return embeddings


class TransformerBlock(nn.Module):
    """
    Transformer Block pentru text (identic cu Image Encoder).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass cu opțional key_padding_mask.

        Args:
            x: [batch, seq_len, dim]
            key_padding_mask: [batch, seq_len] — True pentru padding tokens
        """
        # Attention block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.drop1(attn_out)

        # MLP block
        x = x + self.mlp(self.norm2(x))

        return x


# ═════════════════════════════════════════════════════════════════════════════
# TEST — verificare că Text Encoder funcționează
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Test TextEncoder ===\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CREEAZĂ MODELUL
    # ─────────────────────────────────────────────────────────────────────────
    model = TextEncoder(
        vocab_size=30522,
        max_seq_len=77,
        embed_dim=256,
        depth=4,
        num_heads=4,
        output_dim=256,
        pooling='mean'  # Sau 'cls'
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
    seq_len = 20  # Lungime variabilă (în practică vine din tokenizer)

    # Fake token IDs (în realitate vin din tokenizer)
    fake_input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)

    # Fake attention mask (1 = token real, 0 = padding)
    fake_attention_mask = torch.ones(batch_size, seq_len).to(device)
    # Simulează padding pe ultimele 5 tokeni
    fake_attention_mask[:, -5:] = 0

    model.eval()
    with torch.no_grad():
        embeddings = model(fake_input_ids, fake_attention_mask)

    print(f"Input shape:  {fake_input_ids.shape}")  # [8, 20]
    print(f"Output shape: {embeddings.shape}")  # [8, 256]
    print(f"Output sample:\n{embeddings[0][:10]}...\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. VERIFICARE NORMĂ L2
    # ─────────────────────────────────────────────────────────────────────────
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"L2 norms: {norms[:5]}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. NUMĂR PARAMETRI
    # ─────────────────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParametri totali: {total_params:,}")
    print(f"Parametri antrenabili: {trainable_params:,}")

    print("\nTextEncoder funcționează!")