"""
Step 3: Fine-tune MedSigLIP MULTI-TASK cu DUAL PROMPTS
Optimizat pentru RTX 3090 24GB - gradient accumulation + memory management

Arhitectura:
    prompt_a → emb_a ─┐
                      ├─→ merged = normalize((emb_a + emb_b) / 2)
    prompt_b → emb_b ─┘

                              ┌─→ Contrastive Loss(image_emb, merged_text_emb)
    Imagine → image_emb ──────┤─→ Severity Head → prezice severity (MSE)
                              └─→ Cls Head → prezice boala (CrossEntropy)

    total_loss = contrastive + λ_sev * severity + λ_cls * classification

Rulare:
    python -m src.pipelines.medsiglip.train_medsiglip
"""

import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ...datasets.oct5k_medsiglip import get_medsiglip_loaders
from ...losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from ...utils.seed import set_seed, SEED


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

class Config:
    model_path = "models/medsiglip-448"

    splits_dir = "data/oct5k/splits"
    split_json = "data/oct5k/medgemma_prompts_split.json"
    severity_json = "data/oct5k/severity_scores.json"

    # === MEMORY OPTIMIZATION ===
    # batch real in GPU = 8, dar acumulam 2 pasi = 16 efectiv
    bs = 8
    grad_accum_steps = 2    # effective batch = bs * grad_accum = 16

    epochs = 30
    warmup = 3
    patience = 8
    grad_clip = 1.0
    min_delta = 0.001

    vision_lr = 1e-6
    text_lr = 1e-5
    head_lr = 1e-4
    wd = 0.01
    min_lr = 1e-7

    lambda_sev = 0.3
    lambda_cls = 0.3

    max_scale = 4.6052

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.cuda.is_available()
    workers = 0

    save_dir = "experiments/medsiglip_pipeline"
    resume = "experiments/medsiglip_pipeline/ckpts/last.pth"

cfg = Config()
os.makedirs(f"{cfg.save_dir}/ckpts", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MODEL MULTI-TASK CU DUAL PROMPT MERGE
# ═══════════════════════════════════════════════════════════════════════

class MedSigLIPMultiTask(nn.Module):

    def __init__(self, model_path, num_classes=4):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )

        init_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

        emb_dim = self.model.config.vision_config.hidden_size

        # severity head: image_emb → severity (0-1)
        self.severity_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # classification head: image_emb → disease class
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.model.parameters())
        heads = total - backbone - 1
        print(f"  MedSigLIP Multi-Task:")
        print(f"    Backbone: {backbone:,} | Heads: {heads:,} | Total: {total:,}")
        print(f"    Embedding dim: {emb_dim}")
        print(f"    Effective batch size: {cfg.bs} x {cfg.grad_accum_steps} = {cfg.bs * cfg.grad_accum_steps}")

    def encode_image(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]
        return F.normalize(out, p=2, dim=-1)

    def forward(self, pixel_values, ids_a, mask_a, ids_b, mask_b):
        img_emb = self.encode_image(pixel_values)
        emb_a = self.encode_text(ids_a, mask_a)
        emb_b = self.encode_text(ids_b, mask_b)

        merged_txt = F.normalize((emb_a + emb_b) / 2, p=2, dim=-1)

        sev_pred = self.severity_head(img_emb).squeeze(-1)
        cls_logits = self.cls_head(img_emb)

        return img_emb, merged_txt, self.logit_scale, sev_pred, cls_logits


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

def build_optimizer(model):
    vision, text, heads = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "severity_head" in name or "cls_head" in name or "logit_scale" in name:
            heads.append(p)
        elif "vision" in name or "visual" in name:
            vision.append(p)
        elif "text" in name:
            text.append(p)
        else:
            heads.append(p)

    groups = [
        {"params": vision, "lr": cfg.vision_lr, "name": "vision"},
        {"params": text, "lr": cfg.text_lr, "name": "text"},
        {"params": heads, "lr": cfg.head_lr, "name": "heads"},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]

    for g in groups:
        print(f"    {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    return torch.optim.AdamW(groups, weight_decay=cfg.wd)


# ═══════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_all(model, loader):
    model.eval()
    all_img, all_txt, all_lbl = [], [], []
    all_sev_pred, all_sev_true = [], []
    all_cls_pred, all_cls_true = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        ids_a = batch["input_ids_a"].to(cfg.device)
        mask_a = batch["attention_mask_a"].to(cfg.device)
        ids_b = batch["input_ids_b"].to(cfg.device)
        mask_b = batch["attention_mask_b"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, _, sp, cl = model(pv, ids_a, mask_a, ids_b, mask_b)

        # mutam imediat pe CPU sa eliberam VRAM
        all_img.append(ie.cpu())
        all_txt.append(te.cpu())
        all_lbl.append(batch["label"])
        all_sev_pred.append(sp.cpu())
        all_sev_true.append(batch["severity"])
        all_cls_pred.append(cl.argmax(1).cpu())
        all_cls_true.append(batch["label"])

        # eliberam referintele GPU
        del pv, ids_a, mask_a, ids_b, mask_b, ie, te, sp, cl

    free_memory()

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)
    labels = torch.cat(all_lbl)

    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    m = {}

    for tag, s in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top = s.topk(k, dim=1)
            correct = sum(labels[i] in labels[top[i]] for i in range(n))
            m[f"{tag}_R@{k}"] = 100.0 * correct / n

    sev_p = torch.cat(all_sev_pred) * 100
    sev_t = torch.cat(all_sev_true) * 100
    m["sev_mae"] = (sev_p - sev_t).abs().mean().item()

    cp = torch.cat(all_cls_pred)
    ct = torch.cat(all_cls_true)
    m["cls_acc"] = (cp == ct).float().mean().item() * 100

    return m


# ═══════════════════════════════════════════════════════════════════════
# TRAIN EPOCH (cu gradient accumulation)
# ═══════════════════════════════════════════════════════════════════════

def run_train(model, loader, contrastive_loss, opt, scaler, ep):
    model.train()
    run_loss, run_c, run_s, run_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0

    cls_crit = nn.CrossEntropyLoss()
    sev_crit = nn.MSELoss()

    opt.zero_grad()  # zero la inceput, nu per batch

    pbar = tqdm(loader, desc=f"Ep {ep+1}/{cfg.epochs} [train]")
    for step, batch in enumerate(pbar):
        pv = batch["pixel_values"].to(cfg.device)
        ids_a = batch["input_ids_a"].to(cfg.device)
        mask_a = batch["attention_mask_a"].to(cfg.device)
        ids_b = batch["input_ids_b"].to(cfg.device)
        mask_b = batch["attention_mask_b"].to(cfg.device)
        labels = batch["label"].to(cfg.device)
        severity = batch["severity"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale, sp, cl = model(pv, ids_a, mask_a, ids_b, mask_b)
            lc = contrastive_loss(ie, te, scale)
            ls = sev_crit(sp, severity)
            lcl = cls_crit(cl, labels)
            loss = lc + cfg.lambda_sev * ls + cfg.lambda_cls * lcl

            # impartim la grad_accum pt media corecta
            loss_scaled = loss / cfg.grad_accum_steps

        scaler.scale(loss_scaled).backward()

        # actualizam weights doar la fiecare grad_accum_steps pasi
        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            with torch.no_grad():
                model.logit_scale.clamp_(0, cfg.max_scale)

        with torch.no_grad():
            i2t, t2i = contrastive_accuracy(ie.detach(), te.detach())

        run_loss += loss.item()
        run_c += lc.item()
        run_s += ls.item()
        run_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        # eliberam referinte GPU
        del pv, ids_a, mask_a, ids_b, mask_b, labels, severity
        del ie, te, sp, cl, loss, loss_scaled, lc, ls, lcl

        pbar.set_postfix(
            L=f"{run_loss/steps:.3f}",
            C=f"{run_c/steps:.3f}",
            S=f"{run_s/steps:.3f}",
            CL=f"{run_cl/steps:.3f}",
            i2t=f"{sum_i2t/steps:.0f}%",
        )

    free_memory()

    return {
        "loss": run_loss/steps, "loss_c": run_c/steps,
        "loss_s": run_s/steps, "loss_cl": run_cl/steps,
        "i2t": sum_i2t/steps, "t2i": sum_t2i/steps,
    }


# ═══════════════════════════════════════════════════════════════════════
# VAL EPOCH
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_val(model, loader, contrastive_loss):
    model.eval()
    run_loss, run_c, run_s, run_cl = 0, 0, 0, 0
    sum_i2t, sum_t2i = 0, 0
    steps = 0

    cls_crit = nn.CrossEntropyLoss()
    sev_crit = nn.MSELoss()

    for batch in tqdm(loader, desc="  Val", leave=False):
        pv = batch["pixel_values"].to(cfg.device)
        ids_a = batch["input_ids_a"].to(cfg.device)
        mask_a = batch["attention_mask_a"].to(cfg.device)
        ids_b = batch["input_ids_b"].to(cfg.device)
        mask_b = batch["attention_mask_b"].to(cfg.device)
        labels = batch["label"].to(cfg.device)
        severity = batch["severity"].to(cfg.device)

        with autocast(cfg.device, enabled=cfg.amp):
            ie, te, scale, sp, cl = model(pv, ids_a, mask_a, ids_b, mask_b)
            lc = contrastive_loss(ie, te, scale)
            ls = sev_crit(sp, severity)
            lcl = cls_crit(cl, labels)
            loss = lc + cfg.lambda_sev * ls + cfg.lambda_cls * lcl

        i2t, t2i = contrastive_accuracy(ie, te)
        run_loss += loss.item()
        run_c += lc.item()
        run_s += ls.item()
        run_cl += lcl.item()
        sum_i2t += i2t
        sum_t2i += t2i
        steps += 1

        # eliberam referinte GPU
        del pv, ids_a, mask_a, ids_b, mask_b, labels, severity
        del ie, te, sp, cl, loss, lc, ls, lcl

    free_memory()

    return {
        "loss": run_loss/steps, "loss_c": run_c/steps,
        "loss_s": run_s/steps, "loss_cl": run_cl/steps,
        "i2t": sum_i2t/steps, "t2i": sum_t2i/steps,
    }


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def save_plots(hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(ep, hist["train_loss"], label="Train", marker="o", ms=2)
    axes[0, 0].plot(ep, hist["val_loss"], label="Val", marker="o", ms=2)
    axes[0, 0].set(title="Total Loss", xlabel="Epoch")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, hist["train_loss_c"], label="Contrastive", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["train_loss_s"], label="Severity", marker="o", ms=2)
    axes[0, 1].plot(ep, hist["train_loss_cl"], label="Classification", marker="o", ms=2)
    axes[0, 1].set(title="Train Loss Breakdown", xlabel="Epoch")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(ep, hist["I2T_R@1"], label="R@1", marker="o", ms=2)
    axes[0, 2].plot(ep, hist["I2T_R@5"], label="R@5", marker="o", ms=2)
    axes[0, 2].plot(ep, hist["I2T_R@10"], label="R@10", marker="o", ms=2)
    axes[0, 2].set(title="I2T Retrieval", xlabel="Epoch", ylabel="%")
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(ep, hist["cls_acc"], marker="o", ms=2, color="green")
    axes[1, 0].set(title="Classification Accuracy", xlabel="Epoch", ylabel="%")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, hist["sev_mae"], marker="o", ms=2, color="orange")
    axes[1, 1].set(title="Severity MAE (%)", xlabel="Epoch")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(ep, hist["logit_scale"], color="red", marker="o", ms=2)
    axes[1, 2].set(title="Logit Scale", xlabel="Epoch")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/training_curves.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"{'='*70}")
    print("  STEP 3: MEDSIGLIP MULTI-TASK (DUAL PROMPT MERGE)")
    print(f"  prompt_a + prompt_b → mean merge → contrastive")
    print(f"  + severity head + classification head")
    print(f"  OPTIMIZED: bs={cfg.bs} x accum={cfg.grad_accum_steps} = {cfg.bs * cfg.grad_accum_steps} effective")
    print(f"{'='*70}")
    print(f"  Vision LR={cfg.vision_lr} | Text LR={cfg.text_lr} | Head LR={cfg.head_lr}")
    print(f"  Lambda sev={cfg.lambda_sev} | Lambda cls={cfg.lambda_cls}")

    set_seed()

    processor = AutoProcessor.from_pretrained(cfg.model_path)
    train_dl, val_dl, test_dl = get_medsiglip_loaders(processor, cfg)

    if train_dl is None or val_dl is None:
        raise RuntimeError("Train sau val loader lipseste!")

    nc = train_dl.dataset.num_classes
    print(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
    print(f"  Classes: {train_dl.dataset.classes}")

    model = MedSigLIPMultiTask(cfg.model_path, num_classes=nc).to(cfg.device)
    loss_fn = SigLIPLoss()

    print("\n  Optimizer:")
    opt = build_optimizer(model)

    warmup = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup)
    cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    sched = SequentialLR(opt, [warmup, cosine], milestones=[cfg.warmup])
    scaler = GradScaler(cfg.device, enabled=cfg.amp)

    hist_keys = [
        "train_loss", "val_loss",
        "train_loss_c", "train_loss_s", "train_loss_cl",
        "val_loss_c", "val_loss_s", "val_loss_cl",
        "I2T_R@1", "I2T_R@5", "I2T_R@10",
        "T2I_R@1", "T2I_R@5", "T2I_R@10",
        "cls_acc", "sev_mae", "logit_scale", "lr",
    ]
    hist = {k: [] for k in hist_keys}
    best_score, wait, start_ep = 0.0, 0, 0

    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["epoch"] + 1
        best_score = ckpt["best_score"]
        wait = ckpt["wait"]
        hist = ckpt["hist"]
        print(f"  Resumed from epoch {start_ep}, best: {best_score:.1f}")

    print(f"\n{'='*70}")

    for ep in range(start_ep, cfg.epochs):
        # === TRAIN ===
        t = run_train(model, train_dl, loss_fn, opt, scaler, ep)

        # === ELIBERAM MEMORIE INAINTE DE VAL ===
        free_memory()

        # === VAL ===
        v = run_val(model, val_dl, loss_fn)

        # === ELIBERAM MEMORIE INAINTE DE EVAL ===
        free_memory()

        # === EVAL (retrieval + severity + cls) ===
        m = eval_all(model, val_dl)

        free_memory()

        sched.step()

        scale = model.logit_scale.item()
        avg_r1 = (m["I2T_R@1"] + m["T2I_R@1"]) / 2
        score = 0.5 * avg_r1 + 0.25 * m["cls_acc"] + 0.25 * max(0, 100 - m["sev_mae"])

        hist["train_loss"].append(t["loss"]); hist["val_loss"].append(v["loss"])
        hist["train_loss_c"].append(t["loss_c"]); hist["train_loss_s"].append(t["loss_s"])
        hist["train_loss_cl"].append(t["loss_cl"])
        hist["val_loss_c"].append(v["loss_c"]); hist["val_loss_s"].append(v["loss_s"])
        hist["val_loss_cl"].append(v["loss_cl"])
        hist["cls_acc"].append(m["cls_acc"]); hist["sev_mae"].append(m["sev_mae"])
        hist["logit_scale"].append(scale); hist["lr"].append(opt.param_groups[0]["lr"])
        for k in ["I2T_R@1","I2T_R@5","I2T_R@10","T2I_R@1","T2I_R@5","T2I_R@10"]:
            hist[k].append(m[k])

        print(
            f"\nEp {ep+1}: Loss T={t['loss']:.3f} V={v['loss']:.3f} "
            f"[C={t['loss_c']:.3f} S={t['loss_s']:.3f} CL={t['loss_cl']:.3f}]"
        )
        print(
            f"  R@1={avg_r1:.1f}% | Cls={m['cls_acc']:.1f}% | "
            f"SevMAE={m['sev_mae']:.1f}% | Score={score:.1f}"
        )

        ckpt_data = {
            "epoch": ep, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "best_score": best_score,
            "wait": wait, "hist": hist,
            "num_classes": nc, "classes": train_dl.dataset.classes,
        }

        if score > best_score + cfg.min_delta:
            best_score = score
            wait = 0
            print(f"  ★ Best: {best_score:.1f}")
            torch.save(ckpt_data, f"{cfg.save_dir}/ckpts/best.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        torch.save(ckpt_data, f"{cfg.save_dir}/ckpts/last.pth")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

    # save final in checkpoints/
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/medsiglip_final.pth")

    pd.DataFrame(hist).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(hist)

    print(f"\n{'='*70}")
    print(f"  DONE! Best Score: {best_score:.1f}")
    print(f"  (50% R@1 + 25% Cls + 25% Severity)")
    print(f"  Final saved: checkpoints/medsiglip_final.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()