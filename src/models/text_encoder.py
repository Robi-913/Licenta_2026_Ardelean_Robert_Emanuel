import torch
import torch.nn as nn


class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        max_len=77,
        embed_dim=256,
        depth=4,
        heads=4,
        out_dim=256,
        pool="mean",
        drop=0.05,
    ):
        super().__init__()
        self.pool = pool

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(drop)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

        self._init_weights()

        print(
            f"TextEncoder: vocab={vocab_size}, dim={embed_dim}, "
            f"depth={depth}, heads={heads}, out={out_dim}, pool={pool}"
        )

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.apply(self._init_single)

    def _init_single(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _pool_output(self, x, mask):
        if self.pool == "cls":
            return x[:, 0]

        if self.pool == "max":
            return x.max(dim=1)[0]

        if mask is not None:
            expanded = mask.unsqueeze(-1).expand(x.size())
            total = expanded.sum(dim=1).clamp(min=1e-9)
            return (x * expanded).sum(dim=1) / total

        return x.mean(dim=1)

    def forward(self, input_ids, attention_mask=None):
        b, seq_len = input_ids.shape

        pos = torch.arange(seq_len, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(b, -1)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)

        pooled = self._pool_output(x, attention_mask)
        return self.proj(pooled)