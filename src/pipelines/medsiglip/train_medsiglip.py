import gc
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoProcessor

import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.datasets.oct5k_medsiglip import make_loaders, OCT5kDataset, collate_oct5k
from src.losses.siglip_loss import SigLIPLoss, contrastive_accuracy
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


class Config:
    model_path = "models/medsiglip-448"
    splits_dir = "data/oct5k/splits_v3"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    severity_json = "data/oct5k/severity_scores_v2.json"
    save_dir = "experiments/medsiglip_v15"
    resume = "experiments/medsiglip_v15/ckpts/last.pth"

    batch_size = 8
    bs = 8
    accum = 8
    epochs = 35
    warmup = 3  # epoci de warmup liniar inainte de cosine annealing
    patience = 10  # cate epoci asteptam fara imbunatatire inainte de early stopping
    grad_clip = 1.0  # clip gradients pt stabilitate
    min_delta = 0.001  # imbunatatire minima considerata progres

    # Learning rates diferentiate pe grupuri de parametri
    head_lr = 1e-4  # capetele de task (cls, sev)
    fusion_lr = 1e-4  # CrossAttentionFusion — mai mic, e mai sensibil
    lora_lr = 1.5e-4  # parametrii LoRA
    wd = 0.01  # weight decay (nu se aplica pe bias/norm)
    min_lr = 1e-7  # limita inferioara pt cosine annealing

    # Ponderi loss multi-task
    # loss_total = loss_contrastiv + lam_sev * loss_severitate + lam_cls * loss_clasificare
    lam_contrastiv = 2.0  # explicit weight pe contrastiv
    lam_sev = 0.2
    lam_cls = 0.3

    # BBox oversampling — imaginile cu leziuni adnotate sunt trase de 3x mai des
    bbox_weight = 3.0

    # Scorul compozit: 50% retrieval + 25% clasificare + 25% severitate
    score_w_retrieval = 0.50
    score_w_cls = 0.25
    score_w_sev = 0.25

    max_logit_scale = 3.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()  # Automatic Mixed Precision
    workers = 0

    @property
    def effective_batch(self):
        return self.batch_size * self.accum

    @property
    def ckpt_dir(self):
        return f"{self.save_dir}/ckpts"


cfg = Config()
os.makedirs(cfg.ckpt_dir, exist_ok=True)

def build_train_loader(processor: AutoProcessor) -> tuple[OCT5kDataset, DataLoader]:
    """
    Construieste loader-ul de train cu WeightedRandomSampler.
    Imaginile cu bounding box (leziuni rare) sunt trase de bbox_weight ori mai des
    => modelul vede mai des cazurile dificile fara sa dublam datasetul.
    """
    ds = OCT5kDataset(
        split_csv=f"{cfg.splits_dir}/train.csv",
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=processor,
        mode="train",
    )

    # Greutate default 1.0 pt toate — ridicata la bbox_weight daca are adnotare
    weights = [1.0] * len(ds)
    n_bbox = 0
    if "has_bbox" in ds.df.columns:
        for idx, has_bbox in enumerate(ds.df["has_bbox"]):
            if has_bbox:
                weights[idx] = cfg.bbox_weight
                n_bbox += 1

    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    print(f"  Train: {len(ds)} imagini ({n_bbox} cu bbox, weight={cfg.bbox_weight}x)")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_oct5k,
        drop_last=True,  # evitam batch-uri incomplete la final de epoca
    )
    return ds, loader

def build_optimizer(model: MedSigLIPMultiTask) -> torch.optim.AdamW:
    """
    AdamW cu 3 grupuri de LR: LoRA > Fusion > Task heads.
    Bias-urile si layerele de normalizare NU primesc weight decay
    (standard practice — regularizarea pe ele face mai mult rau decat bine).
    """
    # Initializam 6 liste: decay / no_decay pentru fiecare grup
    lora_d, lora_nd = [], []
    fusion_d, fusion_nd = [], []
    head_d, head_nd = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # parametrii backbone frozen — sarim peste ei

        # Clasificam parametrul in grupul corect dupa nume
        if "lora" in name.lower():
            d_list, nd_list = lora_d, lora_nd
        elif "fusion" in name:
            d_list, nd_list = fusion_d, fusion_nd
        else:
            d_list, nd_list = head_d, head_nd

        # Bias-uri, vectoti 1D, layere norm => fara weight decay
        no_decay = param.ndim <= 1 or "bias" in name or "norm" in name
        (nd_list if no_decay else d_list).append(param)

    param_groups = [
        {"params": lora_d, "lr": cfg.lora_lr, "weight_decay": cfg.wd},
        {"params": lora_nd, "lr": cfg.lora_lr, "weight_decay": 0.0},
        {"params": fusion_d, "lr": cfg.fusion_lr, "weight_decay": cfg.wd},
        {"params": fusion_nd, "lr": cfg.fusion_lr, "weight_decay": 0.0},
        {"params": head_d, "lr": cfg.head_lr, "weight_decay": cfg.wd},
        {"params": head_nd, "lr": cfg.head_lr, "weight_decay": 0.0},
    ]

    # Filtram grupurile goale (ex. daca un grup nu are parametri)
    return torch.optim.AdamW([g for g in param_groups if g["params"]])

def compute_loss(
        model: MedSigLIPMultiTask,
        batch: dict,
        contrastive_fn: SigLIPLoss,
        cls_fn: nn.CrossEntropyLoss,
        sev_fn: nn.SmoothL1Loss,
) -> tuple:
    """
    Forward pass complet + calcul loss multi-task.
    Folosita identic in run_train si run_val

    Returneaza: (loss_total, loss_c, loss_s, loss_cl, img_emb, fused_emb, sev_pred, cls_logits)
    """
    # Mutam totul pe GPU cu non_blocking=True pt transfer asincron
    pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
    ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
    ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
    ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
    mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)
    labels = batch["label"].to(cfg.device, non_blocking=True)
    severity = batch["severity"].to(cfg.device, non_blocking=True)

    # autocast face calculele in float16 unde e sigur => mai rapid, mai putina VRAM
    with autocast(cfg.device, enabled=cfg.use_amp):
        img_emb, emb_a, emb_b, fused_emb, logit_scale, sev_pred, cls_logits = model(
            pv, ia, ma, ib, mb
        )

        # Loss contrastiv calculat din 3 perspective text si mediat
        # => modelul invata sa alinieze imaginea cu promptul A, B, si combinatia lor
        loss_c = (
                         contrastive_fn(img_emb, emb_a, logit_scale)
                         + contrastive_fn(img_emb, emb_b, logit_scale)
                         + contrastive_fn(img_emb, fused_emb, logit_scale)
                 ) / 3

        loss_s = sev_fn(sev_pred, severity)  # SmoothL1 — mai robust la outlieri decat MSE
        loss_cl = cls_fn(cls_logits, labels)  # CrossEntropy pt clasificare 4 clase

        # Ecuatia finala multi-task
        loss = cfg.lam_contrastiv * loss_c + cfg.lam_sev * loss_s + cfg.lam_cls * loss_cl

    return loss, loss_c, loss_s, loss_cl, img_emb, fused_emb, sev_pred, cls_logits

def run_train(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        contrastive_fn: SigLIPLoss,
        cls_fn: nn.CrossEntropyLoss,
        sev_fn: nn.SmoothL1Loss,
        optimizer: torch.optim.AdamW,
        scaler: GradScaler,
        epoch: int,
) -> dict:
    model.train()
    optimizer.zero_grad()

    # Acumulatori pt metrici de epoch
    totals = dict(loss=0.0, loss_c=0.0, loss_s=0.0, loss_cl=0.0, i2t=0.0, t2i=0.0)
    steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
    for step, batch in enumerate(pbar):
        loss, loss_c, loss_s, loss_cl, img_emb, fused_emb, _, _ = compute_loss(
            model, batch, contrastive_fn, cls_fn, sev_fn
        )

        # Gradient accumulation: impartim loss-ul la accum ca sa mentinem scara corecta
        # Efectul: facem un update de weights la fiecare `accum` pasi, nu la fiecare batch
        scaler.scale(loss / cfg.accum).backward()

        is_update_step = (step + 1) % cfg.accum == 0 or (step + 1) == len(loader)
        if is_update_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # evita exploding gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Clampm logit_scale sa nu creasca prea mult => distributie prea ascutita => instabilitate
            with torch.no_grad():
                model.logit_scale.clamp_(0, cfg.max_logit_scale)

        # Acuratete contrastiva pe batch-ul curent (fara grad)
        with torch.no_grad():
            i2t, t2i = contrastive_accuracy(img_emb.detach(), fused_emb.detach())
            # Monitoring: cosine sim medie pe diagonala (perechi corecte)
            cos_sim = (img_emb.detach() * fused_emb.detach()).sum(dim=-1).mean().item()

        totals["cos_sim"] = totals.get("cos_sim", 0.0) + cos_sim
        totals["loss"] += loss.item()
        totals["loss_c"] += loss_c.item()
        totals["loss_s"] += loss_s.item()
        totals["loss_cl"] += loss_cl.item()
        totals["i2t"] += i2t
        totals["t2i"] += t2i
        steps += 1

        pbar.set_postfix(
            L=f"{totals['loss'] / steps:.3f}",
            C=f"{totals['loss_c'] / steps:.3f}",
            S=f"{totals['loss_s'] / steps:.3f}",
            CL=f"{totals['loss_cl'] / steps:.3f}",
        )

    _free_mem()
    return {k: v / steps for k, v in totals.items()}

@torch.no_grad()
def run_val(
        model: MedSigLIPMultiTask,
        loader: DataLoader,
        contrastive_fn: SigLIPLoss,
        cls_fn: nn.CrossEntropyLoss,
        sev_fn: nn.SmoothL1Loss,
) -> dict:
    model.eval()

    totals = dict(loss=0.0, loss_c=0.0, loss_s=0.0, loss_cl=0.0, i2t=0.0, t2i=0.0)
    steps = 0

    for batch in tqdm(loader, desc="  Val", leave=False):
        loss, loss_c, loss_s, loss_cl, img_emb, fused_emb, _, _ = compute_loss(
            model, batch, contrastive_fn, cls_fn, sev_fn
        )
        i2t, t2i = contrastive_accuracy(img_emb, fused_emb)

        totals["loss"] += loss.item()
        totals["loss_c"] += loss_c.item()
        totals["loss_s"] += loss_s.item()
        totals["loss_cl"] += loss_cl.item()
        totals["i2t"] += i2t
        totals["t2i"] += t2i
        steps += 1

    _free_mem()
    return {k: v / steps for k, v in totals.items()}

@torch.no_grad()
def evaluate(model: MedSigLIPMultiTask, loader: DataLoader) -> dict:
    model.eval()

    all_img_embs, all_txt_embs = [], []
    all_labels, all_sev_labels = [], []
    all_cls_preds, all_sev_preds = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ia = batch["input_ids_a"].to(cfg.device, non_blocking=True)
        ma = batch["attention_mask_a"].to(cfg.device, non_blocking=True)
        ib = batch["input_ids_b"].to(cfg.device, non_blocking=True)
        mb = batch["attention_mask_b"].to(cfg.device, non_blocking=True)

        with autocast(cfg.device, enabled=cfg.use_amp):
            img_emb, _, _, fused_emb, _, sev_pred, cls_logits = model(pv, ia, ma, ib, mb)

        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(fused_emb.cpu())
        all_labels.append(batch["label"])
        all_sev_preds.append(sev_pred.cpu())
        all_sev_labels.append(batch["severity"])
        all_cls_preds.append(cls_logits.argmax(1).cpu())

    _free_mem()

    img_embs = torch.cat(all_img_embs)
    txt_embs = torch.cat(all_txt_embs)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_cls_preds)
    sev_pred_pct = torch.cat(all_sev_preds) * 100
    sev_label_pct = torch.cat(all_sev_labels) * 100

    sim = img_embs @ txt_embs.T
    n = sim.shape[0]

    metrics = {}

    for tag, sim_mat in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top_k = sim_mat.topk(k, dim=1)
            hits = sum(labels[i] in labels[top_k[i]] for i in range(n))
            metrics[f"{tag}_R@{k}"] = 100.0 * hits / n

    metrics["sev_mae"] = (sev_pred_pct - sev_label_pct).abs().mean().item()

    correct_mask = (preds == labels)
    if correct_mask.sum() > 0:
        metrics["sev_mae_correct"] = (sev_pred_pct[correct_mask] - sev_label_pct[correct_mask]).abs().mean().item()

    metrics["cls_acc"] = correct_mask.float().mean().item() * 100

    return metrics


def compute_score(metrics: dict) -> float:
    """
    Scor compozit care balanceaza cele 3 task-uri:
    50% Avg R@1 (retrieval) + 25% Cls Acc + 25% (100 - SevMAE)
    => mai mare = mai bine
    """
    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2
    sev_acc = max(0.0, 100.0 - metrics["sev_mae"])  # convertim MAE in "acuratete"
    return (
            cfg.score_w_retrieval * avg_r1
            + cfg.score_w_cls * metrics["cls_acc"]
            + cfg.score_w_sev * sev_acc
    )

def _build_checkpoint(
        model, optimizer, scheduler, scaler, epoch, best_score, wait, history, n_classes
) -> dict:
    return {
        "epoch": epoch, "model": model.state_dict(),
        "opt": optimizer.state_dict(), "sched": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_score": best_score,
        "wait": wait, "hist": history, "num_classes": n_classes,
    }


def save_checkpoint(ckpt: dict, filename: str) -> None:
    torch.save(ckpt, f"{cfg.ckpt_dir}/{filename}")


def load_checkpoint(path, model, optimizer, scheduler, scaler, history) -> tuple:
    """Incarca checkpoint si returneaza (start_epoch, best_score, wait)."""
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["sched"])
    scaler.load_state_dict(ckpt["scaler"])
    history.update(ckpt["hist"])
    print(f"  Resumed from epoch {ckpt['epoch'] + 1}, best: {ckpt['best_score']:.1f}")
    return ckpt["epoch"] + 1, ckpt["best_score"], ckpt["wait"]

HISTORY_KEYS = [
    "train_loss", "val_loss", "train_loss_c", "train_loss_s", "train_loss_cl",
    "val_loss_c", "val_loss_s", "val_loss_cl",
    "I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10",
    "cls_acc", "sev_mae", "logit_scale", "lr",
]


def update_history(history: dict, train: dict, val: dict, metrics: dict, logit_scale: float, lr: float) -> None:
    history["train_loss"].append(train["loss"])
    history["val_loss"].append(val["loss"])
    history["train_loss_c"].append(train["loss_c"])
    history["train_loss_s"].append(train["loss_s"])
    history["train_loss_cl"].append(train["loss_cl"])
    history["val_loss_c"].append(val["loss_c"])
    history["val_loss_s"].append(val["loss_s"])
    history["val_loss_cl"].append(val["loss_cl"])
    history["cls_acc"].append(metrics["cls_acc"])
    history["sev_mae"].append(metrics["sev_mae"])
    history["logit_scale"].append(logit_scale)
    history["lr"].append(lr)
    for k in ["I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10"]:
        history[k].append(metrics[k])


def save_plots(history: dict) -> None:
    """Salveaza grafice 2x3 cu curbele de antrenare."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _plot(ax, keys_labels, title, ylabel=""):
        for key, label in keys_labels:
            ax.plot(ep, history[key], label=label, marker="o", ms=2)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend();
        ax.grid(alpha=0.3)

    _plot(axes[0, 0], [("train_loss", "Train"), ("val_loss", "Val")], "Total Loss")
    _plot(axes[0, 1], [
        ("train_loss_c", "Contrastive"),
        ("train_loss_s", "Severity"),
        ("train_loss_cl", "Classification"),
    ], "Train Loss Breakdown")
    _plot(axes[0, 2], [("I2T_R@1", "R@1"), ("I2T_R@5", "R@5"), ("I2T_R@10", "R@10")], "I2T Retrieval", "%")
    _plot(axes[1, 0], [("cls_acc", "Acc")], "Classification Accuracy", "%")
    _plot(axes[1, 1], [("sev_mae", "MAE")], "Severity MAE (%)")
    _plot(axes[1, 2], [("logit_scale", "Scale")], "Logit Scale")

    plt.suptitle(
        f"MedSigLIP v13 | LoRA r=16 | lam_cls={cfg.lam_cls} | bbox_weight={cfg.bbox_weight}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/training_curves.png", dpi=150)
    plt.close()

def _free_mem() -> None:
    """Elibereaza cache VRAM si ruleaza GC — chemat dupa fiecare epoca."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    print("  MEDSIGLIP v15 — LoRA + BBox Oversampling + lam_cls=1.0 + 27B prompts")
    print(f"  split_json   = {cfg.split_json}")
    print(f"  batch efectiv = {cfg.batch_size} x {cfg.accum} = {cfg.effective_batch}")
    print(f"  lam_cls={cfg.lam_cls} | lam_sev={cfg.lam_sev} | bbox_weight={cfg.bbox_weight}")

    set_seed()

    wandb.init(
        project="licenta-medsiglip",
        id="3112iuey",
        resume="must",
        config={
            "version": "v15", "model": cfg.model_path,
            "bs_effective": cfg.effective_batch, "epochs": cfg.epochs,
            "prompts": "MedGemma 27B IT", "lora_r": 16, "lora_alpha": 32,
            "lora_lr": cfg.lora_lr, "lam_cls": cfg.lam_cls,
            "lam_sev": cfg.lam_sev, "bbox_weight": cfg.bbox_weight,
            "dataset": "OCT5k",
        },
    )

    processor = AutoProcessor.from_pretrained(cfg.model_path)

    train_ds, train_loader = build_train_loader(processor)
    _, val_loader, _ = make_loaders(processor, cfg)

    if val_loader is None:
        raise RuntimeError("Val loader lipsa — verifica splits_dir in Config.")

    model = MedSigLIPMultiTask(cfg.model_path, n_classes=train_ds.n_classes).to(cfg.device)

    contrastive_fn = SigLIPLoss()
    cls_fn = nn.CrossEntropyLoss()
    sev_fn = nn.SmoothL1Loss()
    optimizer = build_optimizer(model)

    # Scheduler: warmup liniar (LR creste de la 10% la 100%) urmat de cosine annealing
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup, eta_min=cfg.min_lr)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[cfg.warmup])
    scaler = GradScaler(cfg.device, enabled=cfg.use_amp)

    history = {k: [] for k in HISTORY_KEYS}
    best_score = 0.0
    wait = 0
    start_ep = 0

    if cfg.resume and os.path.exists(cfg.resume):
        start_ep, best_score, wait = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, scaler, history
        )

    for epoch in range(start_ep, cfg.epochs):
        train_m = run_train(model, train_loader, contrastive_fn, cls_fn, sev_fn, optimizer, scaler, epoch)
        val_m = run_val(model, val_loader, contrastive_fn, cls_fn, sev_fn)
        eval_m = evaluate(model, val_loader)
        scheduler.step()

        score = compute_score(eval_m)
        avg_r1 = (eval_m["I2T_R@1"] + eval_m["T2I_R@1"]) / 2
        current_lr = optimizer.param_groups[0]["lr"]

        update_history(history, train_m, val_m, eval_m, model.logit_scale.item(), current_lr)

        print(
            f"\nEpoch {epoch + 1}: "
            f"Loss T={train_m['loss']:.3f} V={val_m['loss']:.3f} "
            f"[C={train_m['loss_c']:.3f} S={train_m['loss_s']:.3f} CL={train_m['loss_cl']:.3f}]"
        )
        print(
            f"  R@1={avg_r1:.1f}% | Cls={eval_m['cls_acc']:.1f}% "
            f"| SevMAE={eval_m['sev_mae']:.1f}% | Score={score:.1f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_m["loss"], "train/loss_c": train_m["loss_c"],
            "train/loss_s": train_m["loss_s"], "train/loss_cl": train_m["loss_cl"],
            "val/loss": val_m["loss"], "val/R@1": avg_r1,
            "train/cos_sim": train_m.get("cos_sim", 0),
            "val/cls_acc": eval_m["cls_acc"], "val/sev_mae": eval_m["sev_mae"],
            "val/score": score, "lr": current_lr,
        })

        ckpt = _build_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_score, wait, history, train_ds.n_classes,
        )

        if score > best_score + cfg.min_delta:
            best_score = score
            wait = 0
            print(f"   New best: {best_score:.1f}")
            save_checkpoint(ckpt, "best.pth")
        else:
            wait += 1
            print(f"  ({wait}/{cfg.patience})")

        save_checkpoint(ckpt, "last.pth")  # salvam si last pt a putea relua oricand

        if wait >= cfg.patience:
            print(f"  Early stopping la epoch {epoch + 1}")
            break

    # Salvari finale
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/medsiglip_v15_final.pth")
    pd.DataFrame(history).to_csv(f"{cfg.save_dir}/training_history.csv", index=False)
    save_plots(history)
    wandb.finish()

    print(f"  DONE! Best Score: {best_score:.1f}")


if __name__ == "__main__":
    main()
