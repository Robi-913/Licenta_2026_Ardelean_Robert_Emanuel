import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder


class SigLIPModel(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=32,
        img_dim=384,
        img_depth=6,
        img_heads=6,
        vocab_size=30522,
        max_len=77,
        txt_dim=256,
        txt_depth=4,
        txt_heads=4,
        txt_pool="mean",
        out_dim=256,
    ):
        super().__init__()
        self.out_dim = out_dim

        self.img_enc = ImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=img_dim,
            depth=img_depth,
            heads=img_heads,
            out_dim=out_dim,
        )

        self.txt_enc = TextEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=txt_dim,
            depth=txt_depth,
            heads=txt_heads,
            out_dim=out_dim,
            pool=txt_pool,
        )

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        img_params = sum(p.numel() for p in self.img_enc.parameters())
        txt_params = sum(p.numel() for p in self.txt_enc.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"\nSigLIP: img={img_params:,} txt={txt_params:,} total={total:,} params")

    def encode_image(self, images):
        emb = self.img_enc(images)
        return F.normalize(emb, p=2, dim=1)

    def encode_text(self, input_ids, attention_mask=None):
        emb = self.txt_enc(input_ids, attention_mask)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, images, input_ids, attention_mask=None):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.logit_scale