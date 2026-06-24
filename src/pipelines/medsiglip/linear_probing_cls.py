"""
Linear Probing — antreneaza DOAR cls_head pe features inghetate.

Folosire:
  python -m src.pipelines.medsiglip.linear_probe

Ce face:
  1. Incarca best.pth din v13
  2. Ingheata TOT (backbone, LoRA, fusion, sev_head, logit_scale)
  3. Reseteaza cls_head cu un MLP mai mare
  4. Antreneaza cls_head singur pe features NORMALIZATE (spatiul contrastiv unde R@1=86%)
  5. Salveaza best_probe.pth

R@1 ramane neatins. Cls ar trebui sa urce semnificativ.
Dureaza 3-5 minute.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.datasets.oct5k_medsiglip import make_loaders, OCT5kDataset, collate_oct5k
from src.utils.seed import set_seed


# ---------- config ----------

class Config:
    experiment_dir = "experiments/medsiglip_v13"
    checkpoint = "experiments/medsiglip_v13/ckpts/best.pth"

    model_path = "models/medsiglip-448"
    splits_dir = "data/oct5k/splits_v3"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    severity_json = "data/oct5k/severity_scores_v2.json"

    # LoRA — IDENTIC cu v12/v13
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # probe
    probe_epochs = 20
    probe_lr = 3e-4
    probe_wd = 0.01
    probe_hidden = 512
    probe_dropout = 0.3

    bs = 32
    bbox_weight = 3.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0


cfg = Config()


# ---------- model (identic cu v12/v13) ----------

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


class MedSigLIPMultiTask(nn.Module):
    def __init__(self, model_path, n_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.backbone = get_peft_model(self.backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none",
        ))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))
        bb = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        self.dim = bb.config.vision_config.hidden_size
        self.sev_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, 256),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, 256),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_classes),
        )
        self.fusion = CrossAttentionFusion(self.dim)

    def encode_image(self, pixel_values):
        out = self.backbone.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return out

    def encode_text(self, input_ids, attention_mask):
        out = self.backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        pooled = self.encode_image(pixel_values)
        img_emb = F.normalize(pooled, p=2, dim=-1)
        ea = self.encode_text(ids_a, mask_a)
        eb = self.encode_text(ids_b, mask_b)
        merged = self.fusion(ea, eb)
        sev = self.sev_head(pooled).squeeze(-1).clamp(0, 1)
        cls = self.cls_head(pooled)
        return img_emb, ea, eb, merged, self.logit_scale, sev, cls


# ---------- train loader ----------

def make_train_loader(proc):
    train_csv = f"{cfg.splits_dir}/train.csv"
    ds = OCT5kDataset(
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
    print(f"  Train: {len(ds)} ({n_bbox} bbox)")
    return ds, DataLoader(
        ds, batch_size=cfg.bs, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_oct5k, drop_last=True,
    )


# ---------- full eval ----------

@torch.no_grad()
def full_eval(model, loader):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []
    all_sp, all_st, all_cp = [], [], []

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
    ct = torch.cat(all_lbl)
    out["cls_acc"] = (cp == ct).float().mean().item() * 100

    avg_r1 = (out["I2T_R@1"] + out["T2I_R@1"]) / 2
    out["avg_r1"] = avg_r1
    out["score"] = 0.5 * avg_r1 + 0.25 * out["cls_acc"] + 0.25 * max(0, 100 - out["sev_mae"])
    return out


# ---------- main ----------

def main():
    print(f"{'=' * 60}")
    print(f"  LINEAR PROBE — cls_head pe features normalizate")
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Probe: {cfg.probe_epochs} epoci, lr={cfg.probe_lr}")
    print(f"{'=' * 60}")

    set_seed()
    proc = AutoProcessor.from_pretrained(cfg.model_path)

    train_ds, train_dl = make_train_loader(proc)
    _, val_dl, test_dl = make_loaders(proc, cfg)

    nc = train_ds.n_classes
    print(f"  Val: {len(val_dl.dataset)} | Classes: {train_ds.classes}")

    # incarca modelul
    model = MedSigLIPMultiTask(cfg.model_path, n_classes=nc).to(cfg.device)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    for k in ckpt["model"]:
        if "cls_head" in k:
            print(k, ckpt["model"][k].shape)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint (score={ckpt['best_score']:.1f})")

    # eval INAINTE
    print("\n  Eval INAINTE de linear probe:")
    before = full_eval(model, val_dl)
    print(f"    R@1={before['avg_r1']:.1f}% | Cls={before['cls_acc']:.1f}% | "
          f"SevMAE={before['sev_mae']:.1f}% | Score={before['score']:.1f}")

    # ingheata TOT
    for p in model.parameters():
        p.requires_grad = False

    # cls_head nou
    dim = model.dim
    model.cls_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, cfg.probe_hidden),
        nn.GELU(),
        nn.Dropout(cfg.probe_dropout),
        nn.Linear(cfg.probe_hidden, cfg.probe_hidden // 2),
        nn.GELU(),
        nn.Dropout(cfg.probe_dropout / 2),
        nn.Linear(cfg.probe_hidden // 2, nc),
    ).to(cfg.device)

    for p in model.cls_head.parameters():
        p.requires_grad = True

    n_params = sum(p.numel() for p in model.cls_head.parameters())
    print(f"\n  Cls head nou: {n_params:,} params")

    opt = torch.optim.AdamW(model.cls_head.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_wd)
    sched = CosineAnnealingLR(opt, T_max=cfg.probe_epochs, eta_min=1e-5)
    cls_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_ep = 0

    print(f"\n  Training cls_head ({cfg.probe_epochs} epoci)...\n")

    for ep in range(cfg.probe_epochs):
        model.train()
        tot_loss, tot_ok, tot_n = 0, 0, 0

        for batch in tqdm(train_dl, desc=f"  Probe {ep+1}/{cfg.probe_epochs}", leave=False):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            labels = batch["label"].to(cfg.device, non_blocking=True)

            # features normalizate (spatiul contrastiv) — inghetate
            with torch.no_grad():
                with autocast(cfg.device, enabled=cfg.amp):
                    pooled = F.normalize(model.encode_image(pv), p=2, dim=-1)

            # doar cls_head primeste gradient
            with autocast(cfg.device, enabled=cfg.amp):
                logits = model.cls_head(pooled)
                loss = cls_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.item() * len(labels)
            tot_ok += (logits.argmax(1) == labels).sum().item()
            tot_n += len(labels)

        sched.step()

        # val — tot pe features normalizate
        model.eval()
        val_ok, val_n = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
                labels = batch["label"].to(cfg.device, non_blocking=True)
                with autocast(cfg.device, enabled=cfg.amp):
                    pooled = F.normalize(model.encode_image(pv), p=2, dim=-1)
                    logits = model.cls_head(pooled)
                val_ok += (logits.argmax(1) == labels).sum().item()
                val_n += len(labels)

        train_acc = 100.0 * tot_ok / tot_n
        val_acc = 100.0 * val_ok / val_n
        train_loss = tot_loss / tot_n

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = ep + 1
            torch.save(model.state_dict(), f"{cfg.experiment_dir}/ckpts/best_probe.pth")
            marker = f"  ★ Best: {best_acc:.1f}%"

        print(f"  Ep {ep+1}: Loss={train_loss:.4f} | Train={train_acc:.1f}% | Val={val_acc:.1f}%{marker}")

    # eval complet cu best probe
    model.load_state_dict(
        torch.load(f"{cfg.experiment_dir}/ckpts/best_probe.pth", map_location=cfg.device, weights_only=False)
    )

    print(f"\n  Eval DUPA linear probe (best ep {best_ep}):")
    after = full_eval(model, val_dl)
    print(f"    R@1={after['avg_r1']:.1f}% | Cls={after['cls_acc']:.1f}% | "
          f"SevMAE={after['sev_mae']:.1f}% | Score={after['score']:.1f}")

    if test_dl is not None:
        print(f"\n  Eval pe TEST SET:")
        test_m = full_eval(model, test_dl)
        print(f"    R@1={test_m['avg_r1']:.1f}% | Cls={test_m['cls_acc']:.1f}% | "
              f"SevMAE={test_m['sev_mae']:.1f}% | Score={test_m['score']:.1f}")

    torch.save(model.state_dict(), f"{cfg.experiment_dir}/ckpts/final_with_probe.pth")

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Cls: {before['cls_acc']:.1f}% -> {after['cls_acc']:.1f}% (+{after['cls_acc'] - before['cls_acc']:.1f}%)")
    print(f"  R@1: {before['avg_r1']:.1f}% -> {after['avg_r1']:.1f}% (neschimbat)")
    print(f"  Score: {before['score']:.1f} -> {after['score']:.1f}")
    print(f"  Saved: {cfg.experiment_dir}/ckpts/final_with_probe.pth")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()