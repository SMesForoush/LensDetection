"""Shared fine-tuning machinery for the pretrained lens-finder pipelines.

Mirrors train.py so results are directly comparable with LensNet:
- same data sources and stratified 70/20/10 split (datasets/sources.py), including mock-only runs
- LensDatasetMemoryEfficient (inpainting + outlier clipping) as the base dataset
- class-weighted cross-entropy with label smoothing, AdamW, AMP, gradient clipping
- model selection on validation F2 subject to a minimum specificity, early stopping

Differences from train.py:
- the decision threshold is tuned on the *validation* set and then applied to the test set
- pretrained backbones get their own (lower) learning rate; frozen BatchNorm layers stay in eval mode
- checkpoints store everything needed to rebuild the model and its input transform (see pipelines/load.py)

Models may return 1 logit (sigmoid models: TEGLIE, DeepLens) or 2 logits (GraViT). A single logit z is
mapped to the two-class logits [0, z], for which softmax(.)[1] == sigmoid(z) and cross-entropy equals
binary cross-entropy, so one training loop serves all three models.
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.base import LensDatasetMemoryEfficient  # noqa: E402
from datasets.sources import add_data_arguments, collect_from_args, split_files  # noqa: E402
from datasets.transforms import TransformedDataset  # noqa: E402


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def build_arg_parser(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = add_data_arguments(p.add_argument_group("data (same sources and split as train.py)"))
    d.add_argument("--output_dir", required=True, help="Output directory for checkpoints and results")
    d.add_argument("--bands", help="Comma-separated base-data channel indices fed to the model's input slots")
    d.add_argument("--resize_mode", choices=["resize", "center_crop"], default="resize",
                   help="How the 300x300 cutouts are brought to the model's input size")
    d.add_argument("--augment", action="store_true", help="Random flips + 90 degree rotations on the training set")

    t = p.add_argument_group("training")
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--learning_rate", type=float, default=1e-4, help="LR of the pretrained backbone")
    t.add_argument("--head_learning_rate", type=float, help="LR of the classification head (default: learning_rate)")
    t.add_argument("--weight_decay", type=float, default=1e-2)
    t.add_argument("--warmup_epochs", type=int, default=1, help="Linear LR warm-up before cosine decay")
    t.add_argument("--min_lr_ratio", type=float, default=0.01, help="Final LR as a fraction of the initial LR")
    t.add_argument("--label_smoothing", type=float, default=0.1)
    t.add_argument("--lens_weight_multiplier", type=float, default=1.0)
    t.add_argument("--grad_clip", type=float, default=1.0, help="0 disables clipping")
    t.add_argument("--accumulation_steps", type=int, default=1)
    t.add_argument("--min_specificity", type=float, default=0.4, help="Min val specificity for model selection")
    t.add_argument("--early_stop_patience", type=int, default=10)
    t.add_argument("--checkpoint_freq", type=int, default=0, help="Also save every N epochs (0 = off)")
    t.add_argument("--seed", type=int, default=42)

    s = p.add_argument_group("system")
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--multi_gpu", action="store_true")
    s.add_argument("--num_workers", type=int, default=8)
    s.add_argument("--prefetch_factor", type=int, default=4)
    s.add_argument("--mixed_precision", action="store_true", default=True)
    s.add_argument("--no-mixed_precision", dest="mixed_precision", action="store_false")
    return p


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(output_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(Path(output_dir) / "finetune.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.info(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, NumPy {np.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
        torch.backends.cudnn.benchmark = True
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    return logger


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def make_loader(paths, labels, transform, args, shuffle):
    base = LensDatasetMemoryEfficient(paths, labels, track_statistics=False)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                  shuffle=shuffle, drop_last=shuffle)  # no size-1 training batches (BatchNorm)
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(TransformedDataset(base, transform), **kwargs)


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------
def two_class_logits(output):
    """(B, 1) sigmoid logit -> (B, 2) logits [0, z]; (B, 2) passes through."""
    if output.shape[1] == 1:
        return torch.cat([torch.zeros_like(output), output], dim=1)
    return output


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze_modules(modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)


def set_frozen_bn_eval(model):
    """Keep BatchNorm layers whose parameters are frozen in eval mode (running stats stay pretrained)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(model, head_params, args):
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head = [p for p in head_params if p.requires_grad]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": args.learning_rate, "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": args.head_learning_rate or args.learning_rate, "name": "head"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    """Linear warm-up for `warmup_epochs`, then cosine decay to min_lr_ratio * initial LR (stepped per epoch)."""
    def factor(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / (args.warmup_epochs + 1)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, grad_clip=0.0, accumulation_steps=1):
    model.train()
    set_frozen_bn_eval(model)
    total_loss, correct, total, skipped = 0.0, 0, 0, 0
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if not torch.isfinite(data).all():
            skipped += 1
            continue
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = two_class_logits(model(data))
            loss = criterion(logits.float(), target) / accumulation_steps
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps * target.size(0)
        correct += logits.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    if skipped:
        print(f"[NOTICE] Skipped {skipped} training batches with inf/nan values")
    return (total_loss / total, 100.0 * correct / total) if total else (0.0, 0.0)


@torch.no_grad()
def predict(model, loader, criterion, device, use_amp):
    """Returns (mean loss, y_true, lens probability)."""
    model.eval()
    total_loss, total, targets, scores = 0.0, 0, [], []
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = two_class_logits(model(data)).float()
        total_loss += criterion(logits, target).item() * target.size(0)
        total += target.size(0)
        targets.append(target.cpu().numpy())
        scores.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_score = np.concatenate(scores) if scores else np.zeros(0)
    y_score[~np.isfinite(y_score)] = 0.5
    return total_loss / max(total, 1), (np.concatenate(targets) if targets else np.zeros(0, int)), y_score


def binary_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    two_classes = len(set(y_true.tolist())) > 1
    return {
        "threshold": float(threshold),
        "acc": 100.0 * (tp + tn) / max(len(y_true), 1),
        "recall": recall,
        "precision": precision,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)) if two_classes else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else 0.0,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def best_f2_threshold(y_true, y_score):
    best_t, best_f2 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f2 = fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = float(t), f2
    return best_t


def save_predictions(path, paths, origins, y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    df = pd.DataFrame({"file_path": paths, "data_source": origins, "true_label": y_true,
                       "predicted_label": y_pred, "lens_probability": y_score,
                       "correct": (y_true == y_pred).astype(int)})
    df["outcome"] = np.select([(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 0),
                               (y_true == 0) & (y_pred == 1)], ["TP", "TN", "FP"], default="FN")
    df.to_csv(path, index_label="sample_id")
    return df


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------
def run_finetuning(args, model_name, model, model_config, head_params, train_transform, logger, extra_info=None):
    """Fine-tune `model` on the base dataset, select on validation F2, evaluate on test and save everything.

    Args:
        model_name: 'gravit' | 'teglie' | 'deeplens' (used by pipelines/load.py to rebuild the model).
        model: pretrained model with frozen parameters already set (requires_grad=False).
        model_config: kwargs that rebuild the architecture (stored in the checkpoint).
        head_params: parameters of the classification head (get head_learning_rate).
        train_transform: ModelInputTransform for training; evaluation uses it without augmentation.
    """
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    eval_transform = train_transform.without_augmentation()

    with open(output_dir / "system_info.json", "w") as f:
        json.dump({"model_name": model_name, "model_config": model_config, "transform": eval_transform.config(),
                   "arguments": vars(args), "python_version": sys.version, "pytorch_version": torch.__version__,
                   "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                   **(extra_info or {})}, f, indent=2, default=str)

    logger.info(f"Collecting files (data_mode={args.data_mode})...")
    paths, labels, origins = collect_from_args(args, logger)
    splits = split_files(paths, labels, origins, args.seed, logger)
    train_loader = make_loader(*splits["train"][:2], train_transform, args, shuffle=True)
    val_loader = make_loader(*splits["val"][:2], eval_transform, args, shuffle=False)
    test_loader = make_loader(*splits["test"][:2], eval_transform, args, shuffle=False)
    logger.info(f"Input transform: {eval_transform.config()} (augment={train_transform.augment})")

    model = model.to(device)
    total, trainable = count_parameters(model)
    logger.info(f"Model '{model_name}': {total:,} parameters, {trainable:,} trainable")
    optimizer = build_optimizer(model, head_params, args)
    for g in optimizer.param_groups:
        logger.info(f"  param group '{g['name']}': {sum(p.numel() for p in g['params']):,} params, peak lr={g['lr']:.1e}")
    scheduler = build_scheduler(optimizer, args)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    train_labels = splits["train"][1]
    counts = np.bincount(train_labels, minlength=2)
    weights = torch.tensor([len(train_labels) / (2 * max(c, 1)) for c in counts], dtype=torch.float)
    weights[1] *= args.lens_weight_multiplier
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    logger.info(f"Class weights: {weights.numpy()}, label smoothing: {args.label_smoothing}")
    scaler = make_grad_scaler(use_amp)

    def checkpoint(epoch, metrics):
        return {"model_name": model_name, "model_config": model_config, "transform": eval_transform.config(),
                "model_state_dict": unwrap(model).state_dict(), "epoch": epoch, "val_metrics": metrics,
                "args": vars(args)}

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_recall", "val_precision",
                               "val_specificity", "val_f1", "val_f2", "val_pr_auc", "val_roc_auc", "lr")}
    best_f2, best_epoch, epochs_no_improve = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp,
                                            args.grad_clip, args.accumulation_steps)
        val_loss, y_val, s_val = predict(model, val_loader, criterion, device, use_amp)
        m = binary_metrics(y_val, s_val)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss),
                           ("val_acc", m["acc"]), ("val_recall", m["recall"]), ("val_precision", m["precision"]),
                           ("val_specificity", m["specificity"]), ("val_f1", m["f1"]), ("val_f2", m["f2"]),
                           ("val_pr_auc", m["pr_auc"]), ("val_roc_auc", m["roc_auc"]), ("lr", lr)):
            history[key].append(value)
        logger.info(f"Epoch {epoch}/{args.epochs} ({time.time() - start:.0f}s) | LR {lr:.2e} | "
                    f"train loss {train_loss:.4f} acc {train_acc:.2f}% | val loss {val_loss:.4f} "
                    f"acc {m['acc']:.2f}% recall {m['recall']:.4f} spec {m['specificity']:.4f} "
                    f"F2 {m['f2']:.4f} ROC-AUC {m['roc_auc']:.4f}")

        torch.save(checkpoint(epoch, m), output_dir / "last_model.pth")
        if args.checkpoint_freq and epoch % args.checkpoint_freq == 0:
            torch.save(checkpoint(epoch, m), output_dir / f"checkpoint_epoch_{epoch}.pth")
        if m["f2"] > best_f2 and m["specificity"] >= args.min_specificity:
            best_f2, best_epoch, epochs_no_improve = m["f2"], epoch, 0
            torch.save(checkpoint(epoch, m), output_dir / "best_model.pth")
            logger.info(f"  -> new best model saved (F2 {m['f2']:.4f}, spec {m['specificity']:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                logger.info(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_path = output_dir / "best_model.pth"
    if not best_path.exists():
        logger.warning(f"No epoch reached specificity >= {args.min_specificity}; using the last epoch as best model")
        best_path = output_dir / "last_model.pth"
        torch.save(torch.load(best_path, map_location="cpu"), output_dir / "best_model.pth")
    ckpt = torch.load(best_path, map_location="cpu")
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Evaluating best model (epoch {ckpt['epoch']})")

    _, y_val, s_val = predict(model, val_loader, criterion, device, use_amp)
    threshold = best_f2_threshold(y_val, s_val)
    _, y_test, s_test = predict(model, test_loader, criterion, device, use_amp)
    results = {"best_epoch": int(ckpt["epoch"]), "threshold_selected_on": "validation",
               "validation": binary_metrics(y_val, s_val, threshold),
               "test": binary_metrics(y_test, s_test, threshold),
               "test_at_0.5": binary_metrics(y_test, s_test, 0.5)}
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    save_predictions(output_dir / "validation_predictions.csv", splits["val"][0], splits["val"][2], y_val, s_val, threshold)
    df = save_predictions(output_dir / "test_predictions.csv", splits["test"][0], splits["test"][2], y_test, s_test, threshold)

    t = results["test"]
    logger.info("=" * 80)
    logger.info(f"TEST RESULTS (threshold {threshold:.2f}, tuned for F2 on validation)")
    for key in ("recall", "specificity", "precision", "f1", "f2", "pr_auc", "roc_auc", "acc"):
        logger.info(f"  {key:12s}: {t[key]:.4f}")
    logger.info(f"  confusion   : {t['confusion_matrix']}")
    for source, g in df.groupby("data_source"):
        logger.info(f"  {source}: {len(g)} samples, accuracy {100 * g['correct'].mean():.2f}%")
    logger.info("=" * 80)
    logger.info(f"Best model saved to {output_dir / 'best_model.pth'}")
    return results
