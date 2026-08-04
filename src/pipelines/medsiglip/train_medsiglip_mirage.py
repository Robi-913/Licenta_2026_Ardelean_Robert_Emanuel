import gc
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from PIL import Image
from peft import LoraConfig, get_peft_model

import torchvision.transforms as T

# MIRAGE
sys.path.insert(0, os.path.abspath("model/mirage"))
from huggingface_hub import PyTorchModelHubMixin
from mirage_hf import MIRAGEWrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.datasets.oct5k_medsiglip import OCT5kDataset
from src.losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from src.utils.seed import set_seed


class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
    def __init__(self, input_size=512, patch_size=32, modalities="bscan", size="base"):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            size=size,
        )


class Config:
    # MIRAGE image encoder
    mirage_model = "model/mirage/MIRAGE-Large"
    mirage_size = "large"
    mirage_dim = 1024
    mirage_input_size = 512

    # MedSigLIP text encoder (inghetat, doar pt text)
    medsiglip_path = "model/medsiglip-448"
    text_dim = 1152

    splits_dir = "data/oct5k/splits_v3"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    severity_json = "data/oct5k/severity_scores_v2.json"

    bs = 128
    accum = 2

    epochs = 30
    warmup = 3
    patience = 8
    grad_clip = 1.0
    min_delta = 0.001

    proj_lr = 5e-4  # projection layer (random init, needs higher lr)
    head_lr = 1e-4
    fusion_lr = 5e-5
    wd = 0.01
    min_lr = 1e-7

    lam_sev = 0.3
    lam_cls = 1.0

    bbox_weight = 3.0
    max_scale = 3.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0

    save_dir = "experiments/medsiglip_v14_mirage"
    resume = None


cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)

mirage_transform_train = T.Compose([
    T.Resize((cfg.mirage_input_size, cfg.mirage_input_size)),
    T.Grayscale(num_output_channels=1),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),  # [0, 1]
])

mirage_transform_eval = T.Compose([
    T.Resize((cfg.mirage_input_size, cfg.mirage_input_size)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
])


class OCT5kMIRAGE(OCT5kDataset):
    """
    Extinde OCT5kDataset: imaginea e preprocedata pt MIRAGE (512x512, grayscale, [0,1]).
    Textul e tokenizat cu MedSigLIP tokenizer (ca inainte).
    """

    def __init__(self, split_csv, split_json, severity_json, processor,
                 img_dirs=None, mode="train"):
        super().__init__(split_csv, split_json, severity_json, processor,
                         img_dirs, mode)
        self.mirage_transform = mirage_transform_train if mode == "train" else mirage_transform_eval

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = self.lbl_map[row["disease"]]

        disk = self._locate(img_path)
        if disk is None:
            disk = row.get("image_disk_path", "")
            if not os.path.exists(disk):
                raise FileNotFoundError(f"Cannot find: {img_path}")

        img = Image.open(disk).convert("RGB")

        # GaussianBlur + AutoCrop (ca in pipeline-ul original)
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = self._auto_crop(img)

        # MIRAGE transform (resize 512, grayscale, to_tensor)
        # flip e inclus in transform pt train
        pixels = self.mirage_transform(img)  # [1, 512, 512]

        # text tokenizat cu MedSigLIP
        pair = self.prompts[img_path]
        ids_a, mask_a = self._tok(pair["a"])
        ids_b, mask_b = self._tok(pair["b"])

        sev = self.sev[img_path] / 100.0

        return {
            "pixel_values": pixels,  # [1, 512, 512] pt MIRAGE
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "label": label,
            "severity": torch.tensor(sev, dtype=torch.float32),
        }


def collate_mirage(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids_a": torch.stack([b["input_ids_a"] for b in batch]),
        "attention_mask_a": torch.stack([b["attention_mask_a"] for b in batch]),
        "input_ids_b": torch.stack([b["input_ids_b"] for b in batch]),
        "attention_mask_b": torch.stack([b["attention_mask_b"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch]),
    }


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, emb_a, emb_b):
        a, b = emb_a.unsqueeze(1), emb_b.unsqueeze(1)
        attn_a, _ = self.attn_a2b(query=a, key=b, value=b)
        attn_b, _ = self.attn_b2a(query=b, key=a, value=a)
        attn_a, attn_b = attn_a.squeeze(1), attn_b.squeeze(1)
        g = self.gate(torch.cat([attn_a, attn_b], dim=-1))
        fused = g * attn_a + (1 - g) * attn_b
        fused = self.norm(fused + emb_a + emb_b)
        fused = fused + self.proj(fused)
        return F.normalize(fused, p=2, dim=-1)


class MIRAGEMultiTask(nn.Module):
    """
    MIRAGE (image) + MedSigLIP (text) + Projection + Fusion + Heads

    MIRAGE: inghetat, feature extractor OCT
    MedSigLIP text: inghetat
    Projection: 768 -> 1152 (trainable, aliniaza spatiile)
    Fusion + cls_head + sev_head: trainable
    """

    def __init__(self, n_classes=4):
        super().__init__()

        # MIRAGE image encoder + LoRA
        print("  Loading MIRAGE-Large...")
        self.mirage = MIRAGEhf.from_pretrained(
            cfg.mirage_model,
            input_size=cfg.mirage_input_size,
            patch_size=32,
            modalities="bscan",
            size=cfg.mirage_size,
        )

        # LoRA pe MIRAGE (adaptare pt datele tale)
        self.mirage.model = get_peft_model(self.mirage.model, LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["qkv", "proj"],
            bias="none",
        ))
        self.mirage.model.print_trainable_parameters()

        # for p in self.mirage.parameters():
        #     p.requires_grad = False
        # self.mirage.eval()

        n_mirage = sum(p.numel() for p in self.mirage.parameters())
        print(f"    MIRAGE params: {n_mirage:,} (LoRA trainable)")

        # MedSigLIP text encoder (INGHETAT)
        print("  Loading MedSigLIP text encoder...")
        self.text_backbone = AutoModel.from_pretrained(
            cfg.medsiglip_path, torch_dtype=torch.float32,
        )
        for p in self.text_backbone.parameters():
            p.requires_grad = False
        self.text_backbone.eval()

        # projection MIRAGE dim -> MedSigLIP text dim
        self.img_proj = nn.Sequential(
            nn.LayerNorm(cfg.mirage_dim),
            nn.Linear(cfg.mirage_dim, cfg.text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.text_dim, cfg.text_dim),
        )

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        # heads pe MIRAGE features (nu pe projected - mai directe)
        self.sev_head = nn.Sequential(
            nn.LayerNorm(cfg.mirage_dim),
            nn.Linear(cfg.mirage_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cfg.mirage_dim),
            nn.Linear(cfg.mirage_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

        self.fusion = CrossAttentionFusion(cfg.text_dim, heads=4, dropout=0.1)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"    Total: {n_total:,} | Trainable: {n_train:,}")
        print(f"    Projection: {cfg.mirage_dim} -> {cfg.text_dim}")

    def encode_image(self, pixel_values):
        tokens = self.mirage({"bscan": pixel_values})
        global_token = tokens[:, -1, :]
        return global_token

    # def encode_image(self, pixel_values):
    #     with torch.no_grad():
    #         tokens = self.mirage({"bscan": pixel_values})
    #     global_token = tokens[:, -1, :]
    #     return global_token

    def encode_text(self, input_ids, attention_mask):
        """MedSigLIP text encode (inghetat)"""
        with torch.no_grad():
            out = self.text_backbone.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        # image: MIRAGE features (768) -> projected (1152)
        mirage_feat = self.encode_image(pixel_values)  # [B, 768]
        img_proj = self.img_proj(mirage_feat)  # [B, 1152]
        img_emb = F.normalize(img_proj, p=2, dim=-1)  # normalizat pt contrastive

        # text: MedSigLIP (inghetat)
        ea = self.encode_text(ids_a, mask_a)
        eb = self.encode_text(ids_b, mask_b)
        merged = self.fusion(ea, eb)

        # heads pe MIRAGE features directe (nu projected)
        sev = self.sev_head(mirage_feat).squeeze(-1).clamp(0, 1)
        cls = self.cls_head(mirage_feat)

        return img_emb, ea, eb, merged, self.logit_scale, sev, cls


def make_train_loader(proc):
    train_csv = f"{cfg.splits_dir}/train.csv"
    ds = OCT5kMIRAGE(
        split_csv=train_csv, split_json=cfg.split_json,
        severity_json=cfg.severity_json, processor=proc, mode="train",
    )
    n_bbox = 0
    if "has_bbox" in ds.df.columns:
        weights = []
        for bb in ds.df["has_bbox"]:
            if bb:
                weights.append(cfg.bbox_weight)
                n_bbox += 1
            else:
                weights.append(1.0)
    else:
        weights = [1.0] * len(ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    print(f"  Train: {len(ds)} ({n_bbox} bbox, weight={cfg.bbox_weight}x)")
    return ds, DataLoader(
        ds, batch_size=cfg.bs, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_mirage, drop_last=True,
    )


def make_eval_loader(proc, split):
    csv = f"{cfg.splits_dir}/{split}.csv"
    if not os.path.exists(csv):
        return None, None
    ds = OCT5kMIRAGE(
        split_csv=csv, split_json=cfg.split_json,
        severity_json=cfg.severity_json, processor=proc, mode="eval",
    )
    dl = DataLoader(
        ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_mirage, drop_last=False,
    )
    return ds, dl


def make_optimizer(model):
    lora_params, lora_nd = [], []
    proj_params, proj_nd = [], []
    fusion_params, fusion_nd = [], []
    head_params, head_nd = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            target_d, target_nd = lora_params, lora_nd
        elif "img_proj" in name:
            target_d, target_nd = proj_params, proj_nd
        elif "fusion" in name:
            target_d, target_nd = fusion_params, fusion_nd
        else:
            target_d, target_nd = head_params, head_nd

        if p.ndim <= 1 or "bias" in name or "norm" in name:
            target_nd.append(p)
        else:
            target_d.append(p)

    groups = [
        {"params": lora_params, "lr": 1e-4, "weight_decay": cfg.wd, "name": "mirage_lora"},
        {"params": lora_nd, "lr": 1e-4, "weight_decay": 0.0, "name": "mirage_lora_nd"},
        {"params": proj_params, "lr": cfg.proj_lr, "weight_decay": cfg.wd, "name": "projection"},
        {"params": proj_nd, "lr": cfg.proj_lr, "weight_decay": 0.0, "name": "projection_nd"},
        {"params": fusion_params, "lr": cfg.fusion_lr, "weight_decay": cfg.wd, "name": "fusion"},
        {"params": fusion_nd, "lr": cfg.fusion_lr, "weight_decay": 0.0, "name": "fusion_nd"},
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.wd, "name": "heads"},
        {"params": head_nd, "lr": cfg.head_lr, "weight_decay": 0.0, "name": "heads_nd"},
    ]
    groups = [g for g in groups if g["params"]]

    print("  Optimizer groups:")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"    {g['name']}: {n:,} params, lr={g['lr']}")

    return torch.optim.AdamW(groups)


def clear_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def eval_all(model, loader):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []
    all_sp, all_st, all_cp, all_ct = [], [], [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, te, _, sp, cl = model(pv, ia, ma, ib, mb)

        all_img.append(ie.cpu())
        all_txt.append(te.cpu())
        all_lbl.append(batch["label"])
        all_sp.append(sp.cpu())
        all_st.append(batch["severity"])
        all_cp.append(cl.argmax(1).cpu())
        all_ct.append(batch["label"])

    clear_mem()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    out = {}
    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            hit = sum(labels[i] in labels[top[i]] for i in range(n))
            out[f"{tag}_R@{k}"] = 100.0 * hit / n

    sp_pct = torch.cat(all_sp) * 100
    st_pct = torch.cat(all_st) * 100
    out["sev_mae"] = (sp_pct - st_pct).abs().mean().item()

    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    out["cls_acc"] = (cp == ct).float().mean().item() * 100

    return out


def run_train(model, loader, c_loss, opt, scaler, ep):
    model.train()
    model.text_backbone.eval()

    tot_l, tot_c, tot_s, tot_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0
    cls_fn = nn.CrossEntropyLoss()
    sev_fn = nn.SmoothL1Loss()
    opt.zero_grad()

    pbar = tqdm(loader, desc=f"Ep {ep + 1}/{cfg.epochs}")
    for step, batch in enumerate(pbar):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)
        severity = batch["severity"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, merged, scale, sp, cl = model(pv, ia, ma, ib, mb)
            lc_a = c_loss(ie, ea, scale)
            lc_b = c_loss(ie, eb, scale)
            lc_m = c_loss(ie, merged, scale)
            lc = (lc_a + lc_b + lc_m) / 3
            ls = sev_fn(sp, severity)
            lcl = cls_fn(cl, labels)
            loss = lc + cfg.lam_sev * ls + cfg.lam_cls * lcl
            loss_div = loss / cfg.accum

        scaler.scale(loss_div).backward()

        if (step + 1) % cfg.accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, cfg.max_scale)

        with torch.no_grad():
            i2t, t2i = contrastive_accuracy(ie.detach(), merged.detach())

        tot_l += loss.item()
        tot_c += lc.item()
        tot_s += ls.item()
        tot_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        del pv, ia, ma, ib, mb, labels, severity
        del ie, ea, eb, merged, sp, cl, loss, loss_div, lc, lc_a, lc_b, lc_m, ls, lcl

        pbar.set_postfix(
            L=f"{tot_l / steps:.3f}", C=f"{tot_c / steps:.3f}",
            S=f"{tot_s / steps:.3f}", CL=f"{tot_cl / steps:.3f}",
            i2t=f"{sum_i2t / steps:.0f}%",
        )

    clear_mem()
    return {
        "loss": tot_l / steps, "loss_c": tot_c / steps,
        "loss_s": tot_s / steps, "loss_cl": tot_cl / steps,
        "i2t": sum_i2t / steps, "t2i": sum_t2i / steps,
    }


@torch.no_grad()
def run_val(model, loader, c_loss):
    model.eval()
    tot_l, tot_c, tot_s, tot_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0
    cls_fn = nn.CrossEntropyLoss()
    sev_fn = nn.SmoothL1Loss()

    for batch in tqdm(loader, desc="  Val", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)
        severity = batch["severity"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, ea, eb, merged, scale, sp, cl = model(pv, ia, ma, ib, mb)
            lc_a = c_loss(ie, ea, scale)
            lc_b = c_loss(ie, eb, scale)
            lc_m = c_loss(ie, merged, scale)
            lc = (lc_a + lc_b + lc_m) / 3
            ls = sev_fn(sp, severity)
            lcl = cls_fn(cl, labels)
            loss = lc + cfg.lam_sev * ls + cfg.lam_cls * lcl

        i2t, t2i = contrastive_accuracy(ie, merged)
        tot_l += loss.item()
        tot_c += lc.item()
        tot_s += ls.item()
        tot_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

    clear_mem()
    return {
        "loss": tot_l / steps, "loss_c": tot_c / steps,
        "loss_s": tot_s / steps, "loss_cl": tot_cl / steps,
        "i2t": sum_i2t / steps, "t2i": sum_t2i / steps,
    }


def main():
    print(f"{'=' * 70}")
    print("  MEDSIGLIP v14 - MIRAGE Image Encoder + MedSigLIP Text")
    print(f"  MIRAGE: {cfg.mirage_model} ({cfg.mirage_dim}d) -> proj -> {cfg.text_dim}d")
    print(f"  bs={cfg.bs} x accum={cfg.accum} = {cfg.bs * cfg.accum} effective")
    print(f"  lam_cls={cfg.lam_cls} | lam_sev={cfg.lam_sev} | bbox_weight={cfg.bbox_weight}")
    print(f"{'=' * 70}")

    set_seed()

    wandb.init(
        project="licenta-medsiglip",
        name="v14-mirage-base",
        config={
            "version": "v14",
            "image_encoder": "MIRAGE-Base (frozen)",
            "text_encoder": "MedSigLIP (frozen)",
            "projection": f"{cfg.mirage_dim} -> {cfg.text_dim}",
            "bs_effective": cfg.bs * cfg.accum,
            "lam_cls": cfg.lam_cls,
            "bbox_weight": cfg.bbox_weight,
        }
    )

    # MedSigLIP processor pt text tokenization
    proc = AutoProcessor.from_pretrained(cfg.medsiglip_path)

    train_ds, train_dl = make_train_loader(proc)
    _, val_dl = make_eval_loader(proc, "val")
    _, test_dl = make_eval_loader(proc, "test")

    nc = train_ds.n_classes
    print(f"  Val: {len(val_dl.dataset)} | Classes: {train_ds.classes}")

    model = MIRAGEMultiTask(n_classes=nc).to(cfg.device)
    loss_fn = SigLIPLoss()

    print("\n  Optimizer:")
    opt = make_optimizer(model)

    w_sched = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    c_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    sched = SequentialLR(opt, [w_sched, c_sched], milestones=[cfg.warmup])
    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    hist_keys = [
        "train_loss", "val_loss",
        "train_loss_c", "train_loss_s", "train_loss_cl",
        "I2T_R@1", "I2T_R@5", "I2T_R@10",
        "T2I_R@1", "T2I_R@5", "T2I_R@10",
        "cls_acc", "sev_mae", "logit_scale", "lr",
    ]
    hist = {k: [] for k in hist_keys}
    best = 0.0
    wait = 0

    for ep in range(cfg.epochs):
        t = run_train(model, train_dl, loss_fn, opt, scaler, ep)
        clear_mem()

        v = run_val(model, val_dl, loss_fn)
        clear_mem()

        m = eval_all(model, val_dl)
        clear_mem()

        sched.step()

        scale = model.logit_scale.item()
        avg_r1 = (m["I2T_R@1"] + m["T2I_R@1"]) / 2
        score = 0.5 * avg_r1 + 0.25 * m["cls_acc"] + 0.25 * max(0, 100 - m["sev_mae"])

        hist["train_loss"].append(t["loss"])
        hist["val_loss"].append(v["loss"])
        hist["train_loss_c"].append(t["loss_c"])
        hist["train_loss_s"].append(t["loss_s"])
        hist["train_loss_cl"].append(t["loss_cl"])
        hist["cls_acc"].append(m["cls_acc"])
        hist["sev_mae"].append(m["sev_mae"])
        hist["logit_scale"].append(scale)
        hist["lr"].append(opt.param_groups[0]["lr"])
        for k in ["I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10"]:
            hist[k].append(m[k])

        print(
            f"\nEp {ep + 1}: Loss T={t['loss']:.3f} V={v['loss']:.3f} "
            f"[C={t['loss_c']:.3f} S={t['loss_s']:.3f} CL={t['loss_cl']:.3f}]"
        )
        print(
            f"  R@1={avg_r1:.1f}% | Cls={m['cls_acc']:.1f}% | "
            f"SevMAE={m['sev_mae']:.1f}% | Score={score:.1f}"
        )

        wandb.log({
            "epoch": ep + 1,
            "train/loss": t["loss"],
            "val/loss": v["loss"],
            "val/R@1": avg_r1,
            "val/cls_acc": m["cls_acc"],
            "val/sev_mae": m["sev_mae"],
            "val/score": score,
            "logit_scale": scale,
        })

        ckpt = {
            "epoch": ep, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "best_score": best,
            "wait": wait, "hist": hist,
            "num_classes": nc, "classes": train_ds.classes,
            "version": "v14",
        }

        if score > best + cfg.min_delta:
            best = score
            wait = 0
            print(f"   Best: {best:.1f}")
            torch.save(ckpt, f"{cfg.save_dir}/ckpts/best.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        torch.save(ckpt, f"{cfg.save_dir}/ckpts/last.pth")

        if wait >= cfg.patience:
            print(f"  Early stopping la epoch {ep + 1}")
            break

    # test
    if test_dl is not None:
        ckpt = torch.load(f"{cfg.save_dir}/ckpts/best.pth", map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print("\n  TEST SET:")
        test_m = eval_all(model, test_dl)
        avg_r1 = (test_m["I2T_R@1"] + test_m["T2I_R@1"]) / 2
        test_score = 0.5 * avg_r1 + 0.25 * test_m["cls_acc"] + 0.25 * max(0, 100 - test_m["sev_mae"])
        print(f"    R@1={avg_r1:.1f}% | Cls={test_m['cls_acc']:.1f}% | "
              f"SevMAE={test_m['sev_mae']:.1f}% | Score={test_score:.1f}")

    wandb.finish()

    print(f"\n{'=' * 70}")
    print(f"  DONE! Best Score: {best:.1f}")
    print(f"  v14: MIRAGE-Base (frozen) + MedSigLIP text (frozen) + projection")
    print(f"  Saved: {cfg.save_dir}/ckpts/best.pth")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
