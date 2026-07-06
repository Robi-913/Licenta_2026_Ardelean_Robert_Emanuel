"""
linear_probe.py — Linear Probing pe cls_head

Ce face:
  1. Incarca best.pth din v13
  2. Ingheata TOT (backbone, LoRA, fusion, sev_head, logit_scale)
  3. Reseteaza cls_head cu un MLP mai adanc (512 hidden)
  4. Antreneaza cls_head SINGUR pe features NORMALIZATE (spatiul contrastiv R@1=86%)
  5. Salveaza final_with_probe.pth

R@1 ramane neatins — nu atingem backbone-ul.
Cls ar trebui sa urce semnificativ. Dureaza 3-5 minute.

Rulare:
    python -m src.pipelines.medsiglip.linear_probe
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
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.datasets.oct5k_medsiglip import make_loaders, OCT5kDataset, collate_oct5k
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


class Config:
    experiment_dir = "experiments/medsiglip_v15/"
    checkpoint = "experiments/medsiglip_v15/ckpts/best.pth"
    model_path = "models/medsiglip-448"
    splits_dir = "data/oct5k/splits_v3"
    split_json = "data/OCT5k/medgemma_prompts_split_v2_27b.json"
    severity_json = "data/oct5k/severity_scores_v2.json"

    # LoRA — identic cu v13, altfel cheile din checkpoint nu se potrivesc
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # Probe head
    probe_epochs = 20
    probe_lr = 3e-4
    probe_wd = 0.01
    probe_hidden = 512   # mai adanc decat head-ul original de 256
    probe_dropout = 0.3

    batch_size = 32
    bs = 32
    bbox_weight = 3.0     # oversampling pt imaginile cu leziuni adnotate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    workers = 0


cfg = Config()


def build_train_loader(processor: AutoProcessor) -> tuple[OCT5kDataset, DataLoader]:
    ds = OCT5kDataset(
        split_csv=f"{cfg.splits_dir}/train.csv",
        split_json=cfg.split_json,
        severity_json=cfg.severity_json,
        processor=processor,
        mode="train",
    )

    # Imaginile cu bbox (leziuni rare) sunt trase mai des din sampler
    weights = [cfg.bbox_weight if has_bbox else 1.0
               for has_bbox in ds.df.get("has_bbox", [False] * len(ds))]
    n_bbox = sum(1 for w in weights if w > 1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    print(f"  Train: {len(ds)} imagini ({n_bbox} cu bbox, weight={cfg.bbox_weight}x)")

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True,
        collate_fn=collate_oct5k, drop_last=True,
    )
    return ds, loader


@torch.no_grad()
def evaluate(model: MedSigLIPMultiTask, loader: DataLoader) -> dict:
    """
    Evalueaza complet: Recall@K, Cls Acc, SevMAE si scorul compozit.
    Folosit atat inainte cat si dupa probe pt a compara.
    """
    model.eval()
    all_img_embs, all_txt_embs, all_labels = [], [], []
    all_sev_preds, all_sev_labels, all_cls_preds = [], [], []

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

    img_embs = torch.cat(all_img_embs)
    txt_embs = torch.cat(all_txt_embs)
    labels = torch.cat(all_labels)
    sim = img_embs @ txt_embs.T
    n = sim.shape[0]

    metrics = {}
    for tag, sim_mat in [("I2T", sim), ("T2I", sim.T)]:
        for k in [1, 5, 10]:
            _, top_k = sim_mat.topk(k, dim=1)
            hits = sum(labels[i] in labels[top_k[i]] for i in range(n))
            metrics[f"{tag}_R@{k}"] = 100.0 * hits / n

    sev_pred_pct  = torch.cat(all_sev_preds)  * 100
    sev_label_pct = torch.cat(all_sev_labels) * 100
    metrics["sev_mae"] = (sev_pred_pct - sev_label_pct).abs().mean().item()
    metrics["cls_acc"] = (torch.cat(all_cls_preds) == labels).float().mean().item() * 100

    avg_r1 = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2
    metrics["avg_r1"] = avg_r1
    metrics["score"]  = 0.5 * avg_r1 + 0.25 * metrics["cls_acc"] + 0.25 * max(0, 100 - metrics["sev_mae"])
    return metrics


def build_probe_head(embed_dim: int, n_classes: int) -> nn.Sequential:
    """
    MLP mai adanc decat head-ul original (256) — doua straturi ascunse cu GELU.
    Antreneaza pe features L2-normalizate din spatiul contrastiv.
    """
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, cfg.probe_hidden),
        nn.GELU(),
        nn.Dropout(cfg.probe_dropout),
        nn.Linear(cfg.probe_hidden, cfg.probe_hidden // 2),
        nn.GELU(),
        nn.Dropout(cfg.probe_dropout / 2),
        nn.Linear(cfg.probe_hidden // 2, n_classes),
    )


def main():
    print(f"  LINEAR PROBE — cls_head pe features normalizate")
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Probe: {cfg.probe_epochs} epoci, lr={cfg.probe_lr}")

    set_seed()

    processor = AutoProcessor.from_pretrained(cfg.model_path)
    train_ds, train_loader = build_train_loader(processor)
    _, val_loader, test_loader = make_loaders(processor, cfg)

    n_classes = train_ds.n_classes
    print(f"  Val: {len(val_loader.dataset)} | Classes: {train_ds.classes}")

    # Incarcam modelul complet din checkpointul v13
    model = MedSigLIPMultiTask(
        cfg.model_path,
        n_classes=n_classes,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    ).to(cfg.device)

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  Checkpoint incarcat (score={ckpt['best_score']:.1f})")

    # Evaluare de baza inainte de probe
    print("\n  Eval INAINTE de linear probe:")
    before = evaluate(model, val_loader)
    print(f"    R@1={before['avg_r1']:.1f}% | Cls={before['cls_acc']:.1f}% | "
          f"SevMAE={before['sev_mae']:.1f}% | Score={before['score']:.1f}")

    # Inghetam TOT — niciun gradient nu trece prin backbone, fusion sau sev_head
    for param in model.parameters():
        param.requires_grad = False

    # Inlocuim cls_head cu unul mai adanc si dezghetam DOAR el
    model.classification_head = build_probe_head(model.embed_dim, n_classes).to(cfg.device)
    for param in model.classification_head.parameters():
        param.requires_grad = True

    n_probe_params = sum(p.numel() for p in model.classification_head.parameters())
    print(f"\n  Probe head nou: {n_probe_params:,} parametri trainable")

    optimizer = torch.optim.AdamW(
        model.classification_head.parameters(),
        lr=cfg.probe_lr,
        weight_decay=cfg.probe_wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.probe_epochs, eta_min=1e-5)
    cls_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n  Training cls_head ({cfg.probe_epochs} epoci)...\n")

    for epoch in range(cfg.probe_epochs):
        model.eval()
        model.classification_head.train()
        tot_loss, n_correct, n_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"  Probe {epoch + 1}/{cfg.probe_epochs}", leave=False):
            pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
            labels = batch["label"].to(cfg.device, non_blocking=True)

            # Extragem features inghetate in spatiul contrastiv (L2-normalizat)
            # torch.no_grad() explicit — backbone e frozen dar evitam si alocarea grafului
            with torch.no_grad():
                with autocast(cfg.device, enabled=cfg.use_amp):
                    pooled_norm = F.normalize(model.encode_image(pv), p=2, dim=-1)

            # Doar classification_head primeste gradient
            with autocast(cfg.device, enabled=cfg.use_amp):
                logits = model.classification_head(pooled_norm)
                loss = cls_loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * len(labels)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total += len(labels)

        scheduler.step()

        # Validare rapida pe cls_acc — fara retrieval (e prea lent per epoca)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(cfg.device, non_blocking=True)
                labels = batch["label"].to(cfg.device, non_blocking=True)
                with autocast(cfg.device, enabled=cfg.use_amp):
                    pooled_norm = F.normalize(model.encode_image(pv), p=2, dim=-1)
                    logits = model.classification_head(pooled_norm)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = 100.0 * n_correct / n_total
        val_acc = 100.0 * val_correct / val_total
        train_loss = tot_loss / n_total

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{cfg.experiment_dir}/ckpts/best_probe.pth")

        marker = f"  ★ Best: {best_val_acc:.1f}%" if improved else ""
        print(f"  Ep {epoch + 1}: Loss={train_loss:.4f} | Train={train_acc:.1f}% | Val={val_acc:.1f}%{marker}")

    # Evaluare completa cu best probe checkpoint
    best_state = torch.load(
        f"{cfg.experiment_dir}/ckpts/best_probe.pth",
        map_location=cfg.device,
        weights_only=False,
    )
    model.load_state_dict(best_state)

    print(f"\n  Eval DUPA linear probe (best epoch {best_epoch}):")
    after = evaluate(model, val_loader)
    print(f"    R@1={after['avg_r1']:.1f}% | Cls={after['cls_acc']:.1f}% | "
          f"SevMAE={after['sev_mae']:.1f}% | Score={after['score']:.1f}")

    if test_loader is not None:
        print(f"\n  Eval pe TEST SET:")
        test_m = evaluate(model, test_loader)
        print(f"    R@1={test_m['avg_r1']:.1f}% | Cls={test_m['cls_acc']:.1f}% | "
              f"SevMAE={test_m['sev_mae']:.1f}% | Score={test_m['score']:.1f}")

    # Salvam starea finala — aceasta e folosita de eval.py si gradio_app
    torch.save(model.state_dict(), f"{cfg.experiment_dir}/ckpts/final_with_probe.pth")

    print(f"\n  DONE!")
    print(f"  Cls:   {before['cls_acc']:.1f}% -> {after['cls_acc']:.1f}% "
          f"(+{after['cls_acc'] - before['cls_acc']:.1f}%)")
    print(f"  R@1:   {before['avg_r1']:.1f}% -> {after['avg_r1']:.1f}% (neschimbat)")
    print(f"  Score: {before['score']:.1f} -> {after['score']:.1f}")
    print(f"  Saved: {cfg.experiment_dir}/ckpts/final_with_probe.pth")


if __name__ == "__main__":
    main()