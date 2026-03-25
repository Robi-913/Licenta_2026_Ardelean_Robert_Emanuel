import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.layers import DropPath


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, path_drop=0.0):
        """
        :param dim: dimensiunea embeddingului (cat de mare e vectorul care reprezinta fiecare patch)
        :param heads: cate capete de atentie (fiecare se uita la altceva in imagine)
        :param mlp_ratio: cat de mare este layerul ascuns din Feed Forward Network fata de dimensiune (FFN mini retea neuronală simpla doar layere liniare)
        :param drop: probabilitatea de dropout (opreste random niste neuroni la antrenare ca să nu faca overfit)
        :param attn_drop: dropout specific pe atentie
        :param path_drop: DropPath, opreste random un bloc intreg (nu doar neuroni individuali)
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        # normalizam pt a lucra pe aceasi scala

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, # dimensiune vector(384)
            num_heads=heads, # cate capete de atentie (6), practic un cap proceseaza 384/6 = 46 dimensiuni
            dropout=attn_drop, # oprire random a unor conexiuni in atentie
            batch_first=True, # aici e dimensiunea tensorului
        )
        # initializam atentia patchurilor, fiecare patch se uita la celelalte patch-uri pt a intelege contextul

        hidden = int(dim * mlp_ratio) # dimensiunea layerului ascuns din ffn (384 * 4 = 1536)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),  # extinde pt mai mult spatiu si mai multe combinatii
            nn.GELU(), # GELU permite modelului sa invete detalii complexe nu corelatii simple
            nn.Dropout(drop), # oprim random neuroni pt a nu avea overfitting
            nn.Linear(hidden, dim), # comprimam inapoi layerele la dimensiunea initiala
            nn.Dropout(drop), # inca un dropout pt acelasi motiv
        )
        # dupa ce atentia a cules detaliile ffn le proceseaza

        self.drop = nn.Dropout(drop)
        self.path_drop = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()
        # unii neuroni devin 0 si daca aveam prea multi de 0 dam dropout la un bloc intreg pt a nu avea blocuri care nu invata nimic

    def forward(self, x):
        normed = self.ln1(x) # normalizam pt aceasi scala
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False) # self-attention, fiecare patch se uita la celelalte patchuri
        x = x + self.path_drop(self.drop(attn_out)) # trecem prin dropout

        x = x + self.path_drop(self.ffn(self.ln2(x))) # trecem prin ffn si dupa prin dropout -> pastram inputul initial + outputul procesat pt a pastra informatia initiala si a adauga detaliile invatate
        return x


class ImageEncoder(nn.Module):

    def __init__(
        self,
        img_size=224, # 224x224
        patch_size=32, # 224/32 = 7 per axa => 7x7 = 49 patchuri
        embed_dim=384, #  dimensiune vector
        depth=6, # blocuri de transformare
        heads=6, # nr capete
        mlp_ratio=4.0, # cu cat extinde ffn
        out_dim=256, # dimensiune vector iesire
        drop=0.0, # dropout
        path_drop=0.0, # DropPath rate
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3, # canele de culoare
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.num_patches # 49
        # aici taiem in patchuri si transforma in vector de nr

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # factorul de init
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim)) # 50 vec de dimensiune embed_dim -> 49 patchuri si un cls token
        self.pos_drop = nn.Dropout(p=drop) # dropout

        drop_rates = torch.linspace(0, path_drop, depth).tolist() # cream 6 valori egale de la 0 la path_drop
        # ex: path_drop = 0.1, depth = 6 -> drop_rates = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        # primele blocuri vor avea mai putin dropout, blocurile finale vor avea mai mult pt a forta sa invete detalii importante si sa nu se bazeze doar pe blocurile initiale
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=0.0,
                path_drop=drop_rates[i],
            )
            for i in range(depth)
        ]) # cram cele 6 blocuri de transformer

        self.final_norm = nn.LayerNorm(embed_dim) # normalizare finala pt a pastra aceasi scala
        self.proj = nn.Linear(embed_dim, out_dim) # layer liniar care comprima de la 384 la 256

        self._init_weights()

        print(
            f"ImageEncoder: {n_patches} patches, dim={embed_dim}, "
            f"depth={depth}, heads={heads}, out={out_dim}"
        )

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_single)
        # initializam cls_token si pos_embed cu valori mici random
        # iar restul modulelor cu functia de init specifica pentru fiecare tip de modul (linear sau layernorm)


    def _init_single(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        # pentru layerul liniar folosim xavier_uniform care e standard pt a avea o distributie buna a valorilor
        # iar pentru layernorm setam bias la 0 si weight la 1 pt a pastra aceasi scala

    def forward(self, x):
        b = x.shape[0] # cate imagini sunt in batch

        x = self.patch_embed(x) # teiem imaginile in patchuri
        # [B, 3, 224, 224] → [B, 49, 384]
        cls = self.cls_token.expand(b, -1, -1) # cls tokenul e definit o data
        x = torch.cat([cls, x], dim=1) # copiem cls tokenul si il adaugam la fiecare batch
        # [B, 1, 384] -> [B, 49, 384] -> [B, 50, 384]
        x = self.pos_drop(x + self.pos_embed) # adaugam embeddingul de pozitie pt a pastra informatia despre pozitii si aplicam dropout

        for blk in self.blocks:
            x = blk(x)
            # trecem prin cele 6 blocuri

        x = self.final_norm(x) # normalizare finala
        return self.proj(x[:, 0]) # luam cls token din fiecare imagine care a acumulat informatia (din cele 6 blocuri de atentie) dupa comprimam vectorul
        # [B, 256]