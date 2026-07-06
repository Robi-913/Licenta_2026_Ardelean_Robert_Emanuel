"""
src/model/medsiglip.py

Definitiile centrale ale arhitecturii MedSigLIP.
Toate clasele de model importa de AICI — nu se mai copiaza cod intre fisiere.

Ierarhie:
    CrossAttentionFusion          — modul de fuziune text (independent)
    MedSigLIPBase                 — backbone LoRA + encode_image / encode_text
      └── MedSigLIPMultiTask      — + sev_head, cls_head, fusion (antrenare principala)
      └── BiomarkerHeadsV5        — + heads per biomarker (frozen backbone)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel



# CROSS ATTENTION FUSION


class CrossAttentionFusion(nn.Module):
    """
    Fuziune bidirectionala intre doua embeddings de text prin cross-attention.

    In loc sa concatenam sau sa facem media celor doua prompturi,
    le lasam sa se "intrebe" reciproc: promptul A (structural, cum arata retina)
    intreaba promptul B (patologic, ce boala are) si invers.
    Un gate invatat decide per-sample cat conteaza fiecare perspectiva.

    :param embed_dim: Dimensiunea vectorilor de embedding — trebuie sa coincida cu backbone-ul.
    :param num_heads: Numarul de capete de atentie (fiecare se uita la altceva in spatiul semantic).
    :param dropout:   Dropout aplicat in atentie si in proiectia finala pt a evita overfitting.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Atentie A->B: promptul structural intreaba ce spune cel patologic
        self.attn_a_to_b = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Atentie B->A: promptul patologic intreaba ce spune cel structural
        self.attn_b_to_a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

        # Gate-ul: primeste concatenarea celor doua rezultate de atentie si produce o masca [0,1]
        # per dimensiune — practic invata "cat de mult conteaza A vs B pentru fiecare feature"
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

        # Proiectie finala dupa fuziune — adauga capacitate de transformare non-liniara
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, emb_structural: torch.Tensor, emb_pathological: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention asteapta [B, seq_len, dim], iar noi avem [B, dim]
        # => adaugam o dimensiune de secventa fake de 1
        a = emb_structural.unsqueeze(1)   # [B, 1, dim]
        b = emb_pathological.unsqueeze(1) # [B, 1, dim]

        # A intreaba B: "ce informatii patologice sunt relevante pentru ce descriu eu?"
        attn_a, _ = self.attn_a_to_b(query=a, key=b, value=b)
        # B intreaba A: "ce informatii structurale sunt relevante pentru ce descriu eu?"
        attn_b, _ = self.attn_b_to_a(query=b, key=a, value=a)

        # Scoatem dimensiunea de secventa fake, revenim la [B, dim]
        attn_a, attn_b = attn_a.squeeze(1), attn_b.squeeze(1)

        # Gate-ul decide per-feature cat conteaza A vs B
        # daca gate_weights[i] = 0.8 => feature-ul i vine 80% din A si 20% din B
        gate_weights = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = gate_weights * attn_a + (1 - gate_weights) * attn_b

        # Conexiune reziduala cu ambele embedinguri originale + normalizare
        # pastreaza informatia initiala si adauga detaliile invatate prin atentie
        fused = self.norm(fused + emb_structural + emb_pathological)
        fused = fused + self.proj(fused) # inca o reziduala dupa proiectie

        # Normalizam L2 la final pt ca folosim dot-product similarity in loss-ul contrastiv
        return F.normalize(fused, p=2, dim=-1)



# BASE CLASS — backbone LoRA + encode_image / encode_text


class MedSigLIPBase(nn.Module):
    """
    Clasa de baza care incarca backbone-ul MedSigLIP cu LoRA aplicat.

    LoRA (Low-Rank Adaptation) ne permite sa fine-tunam un model de miliarde de parametri
    fara sa actualizam toti parametrii. In loc sa modificam W direct, adaugam doua
    matrice mici A si B (cu rank r mic) si actualizam doar A si B: W' = W + alpha/r * B*A
    => de la ~400M parametri antrenabili ajungem la ~3-5M.

    Ofera encode_image si encode_text — folosite de toate clasele derivate.
    Nu are capete de task, nu se instantiaza direct.

    :param model_path:    Calea locala catre modelul MedSigLIP (sau HuggingFace hub).
    :param lora_rank:     Rangul LoRA (r). Valori mai mari = mai multi parametri, mai multa capacitate.
    :param lora_alpha:    Factorul de scalare LoRA. Conventie: alpha = 2 * r.
    :param lora_dropout:  Dropout in straturile LoRA, pt regularizare.
    """

    # Module-urile din backbone pe care aplicam LoRA
    # q/k/v = query/key/value din self-attention, out = proiectia de iesire din attention
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

    def __init__(
        self,
        model_path: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        # Incarcam backbone-ul si aplicam LoRA pe el
        self.backbone = self._build_lora_backbone(model_path, lora_rank, lora_alpha, lora_dropout)
        # Retinem dimensiunea embedding-ului vizual — folosita de toate head-urile
        self.embed_dim: int = self._get_vision_embed_dim()

    def _build_lora_backbone(self, model_path: str, rank: int, alpha: int, dropout: float) -> nn.Module:
        # Incarcam modelul pretrained in float32 (mai stabil pe GPU-urile noastre)
        backbone = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager"  # <-- parametrul adăugat
        )

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=self.LORA_TARGET_MODULES,
            bias="none", # nu adaugam bias extra in straturile LoRA
        )
        # get_peft_model ingheta toti parametrii originali si adauga matricele A si B
        return get_peft_model(backbone, lora_cfg)

    def _get_vision_embed_dim(self) -> int:
        # PEFT inveleste backbone-ul intr-un PeftModel, asa ca trebuie sa coboram
        # la .base_model.model ca sa ajungem la config-ul original al modelului
        base = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        return base.config.vision_config.hidden_size

    def _pool_features(self, model_output) -> torch.Tensor:
        # MedSigLIP poate returna tipuri diferite de output in functie de versiune
        # => verificam ce avem si extragem reprezentarea pooled corecta
        if hasattr(model_output, "pooler_output"):
            return model_output.pooler_output         # [B, dim] — cls token procesat
        elif hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state[:, 0]  # [B, dim] — primul token (cls)
        return model_output # daca e deja tensor, il returnam direct

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Returnam features BRUTE, ne-normalizate — pt ca head-urile de task
        # (clasificare, severitate) au nevoie de valorile absolute, nu de directii
        return self._pool_features(self.backbone.get_image_features(pixel_values=pixel_values))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raw = self._pool_features(
            self.backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )
        # Normalizam L2 textul pt ca loss-ul contrastiv SigLIP lucreaza cu cosine similarity
        # => vectorii trebuie sa fie pe sfera unitara
        return F.normalize(raw, p=2, dim=-1)



# MULTI-TASK MODEL — antrenare principala v13


class MedSigLIPMultiTask(MedSigLIPBase):
    """
    Model multi-task complet: retrieval contrastiv + clasificare boala + regresie severitate.

    Mosteneste MedSigLIPBase pentru backbone si encode_*.
    Adauga pe deasupra: logit_scale, severity_head, classification_head, CrossAttentionFusion.

    Cele 3 task-uri ruleaza simultan pe acelasi backbone — gradientii din toate 3
    actualizeaza impreuna backbone-ul, ceea ce ajuta la generalizare.

    :param model_path:    Calea catre MedSigLIP.
    :param n_classes:     Numarul de clase de boala (4: AMD / DME / DRUSEN / NORMAL).
    :param cls_hidden:    256 = capul simplu din v13; 512 = capul mai adanc din linear_probe.
    :param lora_rank:     Rangul LoRA, pasat mai departe la MedSigLIPBase.
    :param lora_alpha:    Alpha LoRA.
    :param lora_dropout:  Dropout LoRA.
    """

    def __init__(
        self,
        model_path: str,
        n_classes: int = 4,
        cls_hidden: int = 256,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        # Initializam backbone-ul cu LoRA din clasa parinte
        super().__init__(model_path, lora_rank, lora_alpha, lora_dropout)

        # Scara logit pentru loss-ul contrastiv SigLIP
        # Initializata la log(1/0.07) ≈ 2.65, ca in CLIP original
        # Este un parametru invatat — modelul ajusteaza cat de "ascutita" e distributia de similaritate
        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        # Cap de regresie severitate: prezice un scor 0-100% al bolii
        self.severity_head = self._build_severity_head()

        # Cap de clasificare: prezice clasa de boala (AMD / DME / DRUSEN / NORMAL)
        # cls_hidden controleaza daca folosim varianta simpla sau cea mai adanca
        self.classification_head = self._build_classification_head(n_classes, cls_hidden)

        # Modulul de fuziune care combina promptul A (structural) cu promptul B (patologic)
        self.fusion = CrossAttentionFusion(self.embed_dim, num_heads=4, dropout=0.1)

    def _build_severity_head(self) -> nn.Sequential:
        dim = self.embed_dim
        # LayerNorm la inceput stabilizeaza input-ul din backbone
        # Linear 256 => comprimam la un spatiu mai mic inainte de predictie
        # Output: un singur scalar per imagine, clamped ulterior in [0, 1]
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def _build_classification_head(self, n_classes: int, hidden: int) -> nn.Sequential:
        dim = self.embed_dim
        if hidden == 512:
            # Varianta mai adanca folosita in linear_probe / zero_shot
            # Doua straturi ascunse cu GELU (mai smooth decat ReLU) si dropout mai agresiv
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, n_classes),
            )
        # Varianta simpla din v13 — mai putini parametri, mai putine sanse de overfit
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Extragem features de imagine — ne-normalizate pt head-uri
        image_pooled = self.encode_image(pixel_values) # [B, dim]

        # Normalizam separat pt loss-ul contrastiv (dot-product pe sfera unitara)
        image_emb = F.normalize(image_pooled, p=2, dim=-1) # [B, dim]

        # Encodam cele doua prompturi text — deja normalizate L2 din encode_text
        emb_structural   = self.encode_text(input_ids_a, attention_mask_a)   # [B, dim]
        emb_pathological = self.encode_text(input_ids_b, attention_mask_b)   # [B, dim]

        # Fuzionam cele doua embedinguri text prin cross-attention
        # rezultatul e un vector care combina informatia din ambele prompturi
        fused_text_emb = self.fusion(emb_structural, emb_pathological) # [B, dim]

        # Head-urile de task ruleaza pe features brute (image_pooled), nu pe cel normalizat
        # clamp(0, 1) pt ca label-urile de severitate sunt procente normalizate
        severity_pred = self.severity_head(image_pooled).squeeze(-1).clamp(0, 1) # [B]
        class_logits  = self.classification_head(image_pooled)                   # [B, n_classes]

        return image_emb, emb_structural, emb_pathological, fused_text_emb, self.logit_scale, severity_pred, class_logits



# BIOMARKER HEADS V5 — frozen backbone, heads per biomarker


class BiomarkerHeadsV5(nn.Module):
    """
    Capete de detectie biomarkeri cu backbone frozen.

    Strategia: luam backbone-ul deja antrenat din v13 si inghetam toti parametrii lui.
    Antrenam doar head-urile de biomarkeri deasupra features-urilor extrase.
    Asta e eficient pt ca nu reantrenam tot modelul pentru o sarcina auxiliara.

    :param backbone:      Instanta MedSigLIPBase (sau MedSigLIPMultiTask) deja incarcata cu checkpoint v13.
    :param n_biomarkers:  Numarul de biomarkeri de detectat (default 9).
    """

    def __init__(self, backbone: MedSigLIPBase, n_biomarkers: int = 9):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

        # Inghetam backbone-ul complet — niciun parametru din el nu va fi actualizat
        # Asta reduce VRAM si timpii de antrenare semnificativ
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Cate un head independent per biomarker — fiecare invata sa detecteze una
        # ModuleList => PyTorch stie sa le inregistreze corect ca parametri ai modelului
        self.heads = nn.ModuleList([
            self._build_biomarker_head() for _ in range(n_biomarkers)
        ])

    def _build_biomarker_head(self) -> nn.Sequential:
        dim = self.embed_dim
        # Retea mai adanca decat head-ul de severitate — biomarkerii sunt detectii binare
        # mai fine, au nevoie de mai multa capacitate de discriminare
        # Dropout agresiv (0.3) pt ca dataset-ul de biomarkeri e mic
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),   # output: un singur logit per biomarker
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # torch.no_grad() pe backbone — nu vrem sa calculam gradienti prin el
        # economisim VRAM si timp, backbone-ul e oricum frozen
        with torch.no_grad():
            image_features = self.backbone.encode_image(pixel_values) # [B, dim]

        # Fiecare head produce [B, 1], le concatenam pe ultima dimensiune => [B, n_biomarkers]
        logits = torch.cat([head(image_features) for head in self.heads], dim=-1)
        return logits