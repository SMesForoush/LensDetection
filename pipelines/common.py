"""Shared training machinery for every lens-finder pipeline (LensNet, DeepLens, EfficientNet-B0, TEGLIE, GraViT).

- same data sources and split (datasets/sources.py): --train_data prism (default) | real | both; PRISM 70/15/15
  grouped by base lens system, real images by sky region
- LensDatasetMemoryEfficient (inpainting + outlier clipping) as the base dataset
- one training recipe for every from-scratch run (C13 in TODO.md, the `thesis` recipe = the build_arg_parser
  defaults): cross-entropy without class weights or label smoothing, AdamW (lr 1e-3, weight decay 1e-4), 1 epoch of
  linear warm-up then cosine decay, batch 64, at most 50 epochs, D4 flips/rotations on, PyTorch's default init
- model selection and early stopping on the validation loss: the lowest-loss epoch is kept, patience 10
- the decision threshold is tuned on the *validation* set and then applied to the test set
- pretrained runs (--weights, GraViT) use the `finetune` recipe instead (FINETUNE_RECIPE plus each model's own
  overrides, see parse_args_with_recipe): a side experiment, labelled as such in system_info.json
- pretrained backbones may get their own learning rate; frozen BatchNorm layers stay in eval mode
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
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, fbeta_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.base import LensDatasetMemoryEfficient  # noqa: E402
from datasets.quality import dropped_counts, save_report  # noqa: E402
from datasets.sources import add_data_arguments, collect_from_args, resolve_train_data, split_samples  # noqa: E402
from datasets.transforms import (INPUT_PRESETS, THESIS_TRANSFORM, ModelInputTransform, TransformedDataset,  # noqa: E402
                                 parse_bands)


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def build_arg_parser(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = add_data_arguments(p.add_argument_group("data (same sources and split as train.py)"))
    d.add_argument("--output_dir", required=True, help="Output directory for checkpoints and results")
    d.add_argument("--input_preset", choices=INPUT_PRESETS, default="thesis",
                   help="thesis: the one shared input of the thesis experiment for every model (4 bands 0,1,2,3, "
                        "300x300, per-band unit_asinh; datasets/transforms.py::THESIS_TRANSFORM). native: the "
                        "model's own published input (band order, size, scaling), for the pretrained side "
                        "experiment. --bands, --image_size, --resize_mode and the model's scaling options only "
                        "apply with native")
    d.add_argument("--bands", help="Comma-separated base-data channel indices fed to the model's input slots "
                                   "(--input_preset native only; default: the model's published order)")
    d.add_argument("--resize_mode", choices=["resize", "center_crop"], default="resize",
                   help="How the 300x300 cutouts are brought to the model's input size (--input_preset native only)")
    d.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Random flips + 90 degree rotations (the D4 group) on the training set")

    # Defaults = the thesis recipe for from-scratch runs (C13). Pretrained runs override some of them through
    # parse_args_with_recipe (FINETUNE_RECIPE + per-model values); nothing else should call set_defaults on these.
    t = p.add_argument_group("training (defaults: the shared thesis recipe; pretrained runs use the finetune recipe)")
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    t.add_argument("--learning_rate", type=float, default=1e-3,
                   help="Peak LR (of the backbone, if --head_learning_rate is set)")
    t.add_argument("--head_learning_rate", type=float, help="LR of the classification head (default: learning_rate)")
    t.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW (decoupled) weight decay")
    t.add_argument("--warmup_epochs", type=int, default=1, help="Linear LR warm-up before cosine decay")
    t.add_argument("--min_lr_ratio", type=float, default=0.01, help="Final LR as a fraction of the initial LR")
    t.add_argument("--class_weighting", choices=["none", "balanced"], default="none",
                   help="none: plain cross-entropy (PRISM is 1:1). balanced: weight each class by "
                        "n / (2 * n_class) in the training split")
    t.add_argument("--label_smoothing", type=float, default=0.0)
    t.add_argument("--lens_weight_multiplier", type=float, default=1.0,
                   help="Extra factor on the lens-class loss weight (1 = none)")
    t.add_argument("--grad_clip", type=float, default=1.0, help="0 disables clipping")
    t.add_argument("--accumulation_steps", type=int, default=1)
    t.add_argument("--early_stop_patience", type=int, default=10,
                   help="Stop after this many epochs without a lower validation loss")
    t.add_argument("--min_epochs", type=int, default=0,
                   help="Never stop early before this many epochs have been trained")
    t.add_argument("--checkpoint_freq", type=int, default=0, help="Also save every N epochs (0 = off)")
    t.add_argument("--seed", type=int, default=42)

    s = p.add_argument_group("system")
    s.add_argument("--gpu", action="store_true", help="Use a GPU: NVIDIA (CUDA) if present, else Apple Silicon (MPS)")
    s.add_argument("--multi_gpu", action="store_true")
    s.add_argument("--num_workers", type=int, default=8)
    s.add_argument("--prefetch_factor", type=int, default=4)
    s.add_argument("--mixed_precision", action="store_true", default=True)
    s.add_argument("--no-mixed_precision", dest="mixed_precision", action="store_false")
    return p


# The training recipes. `thesis` is the build_arg_parser defaults above (C13), used by every from-scratch run.
# `finetune` is used only when a run starts from pretrained weights (the side experiment): it restores the settings
# the pipelines used before C13, and each model adds its own published fine-tuning values on top (`model_finetune`).
RECIPES = ("thesis", "finetune")
FINETUNE_RECIPE = dict(learning_rate=1e-4, weight_decay=1e-2, epochs=30, label_smoothing=0.1,
                       class_weighting="balanced")


def parse_args_with_recipe(p, argv=None, pretrained=None, model_finetune=None):
    """Parse `argv` with the thesis recipe, or with the finetune recipe when the run is pretrained.

    pretrained: callable(namespace) -> bool, evaluated on a first parse (e.g. `lambda a: bool(a.weights)`).
    model_finetune: the model's own finetune-recipe defaults, applied on top of FINETUNE_RECIPE.
    Arguments given on the command line always win. The chosen recipe is stored as `args.recipe`.
    """
    first, _ = p.parse_known_args(argv)
    recipe = "finetune" if pretrained is not None and pretrained(first) else "thesis"
    if recipe == "finetune":
        p.set_defaults(**{**FINETUNE_RECIPE, **(model_finetune or {})})
    args = p.parse_args(argv)
    args.recipe = recipe
    return args


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


def mps_available():
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def select_device(use_gpu, logger=None):
    """--gpu: an NVIDIA GPU (CUDA) if present, else the Apple Silicon GPU (MPS), else the CPU."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_gpu and mps_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if use_gpu and logger:
            logger.warning("--gpu given but no CUDA or Apple (MPS) GPU is available; running on the CPU")
    if logger:
        logger.info(f"Using device: {device}" + (" (Apple GPU; mixed precision is CUDA-only, so it is off)"
                                                  if device.type == "mps" else ""))
    return device


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def build_input_transform(args, native_config, logger):
    """The training-set ModelInputTransform chosen by --input_preset (evaluation drops the augmentation).

    thesis: ModelInputTransform.thesis_default(), identical for every architecture. --bands and --image_size may be
        left out or set to the thesis values; anything else is an error, so that a run cannot silently mix presets.
    native: `native_config`, the model's published input convention (pretrained side experiment).
    """
    if args.input_preset == "native":
        transform = ModelInputTransform(augment=args.augment, **native_config)
    elif args.input_preset == "thesis":
        conflicts = []
        if args.bands is not None and parse_bands(args.bands) != THESIS_TRANSFORM["bands"]:
            conflicts.append(f"--bands {args.bands}")
        if getattr(args, "image_size", None) not in (None, THESIS_TRANSFORM["image_size"]):
            conflicts.append(f"--image_size {args.image_size}")
        if args.resize_mode != THESIS_TRANSFORM["resize_mode"]:
            conflicts.append(f"--resize_mode {args.resize_mode}")
        if conflicts:
            raise SystemExit(f"{', '.join(conflicts)}: only valid with --input_preset native; the thesis preset fixes "
                             f"the input to {THESIS_TRANSFORM}")
        transform = ModelInputTransform.thesis_default(augment=args.augment)
    else:
        raise ValueError(f"Unknown input preset '{args.input_preset}'")
    logger.info(f"Input preset '{args.input_preset}': {transform.config()}")
    return transform


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


def class_weights(train_labels, class_weighting="none", lens_weight_multiplier=1.0):
    """Cross-entropy class weights [non-lens, lens], or None for the plain (unweighted) loss."""
    if class_weighting == "balanced":
        counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=2)
        weights = [len(train_labels) / (2 * max(c, 1)) for c in counts]
    elif class_weighting == "none":
        weights = [1.0, 1.0]
    else:
        raise ValueError(f"Unknown class weighting '{class_weighting}'")
    weights[1] *= lens_weight_multiplier
    return None if weights == [1.0, 1.0] else torch.tensor(weights, dtype=torch.float)


def build_criterion(train_labels, args, device, logger=None):
    weights = class_weights(train_labels, args.class_weighting, args.lens_weight_multiplier)
    if logger:
        logger.info(f"Loss: cross-entropy, class weights: {'none' if weights is None else weights.numpy()} "
                    f"(--class_weighting {args.class_weighting}), label smoothing: {args.label_smoothing}")
    return nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                               label_smoothing=args.label_smoothing)


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
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "prevalence": float(np.mean(y_true)) if len(y_true) else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)) if two_classes else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else 0.0,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


# Bootstrap 95 % intervals over the evaluated images (C15). Resampling is stratified by class, so every replicate keeps
# the lens and non-lens counts (and the prevalence) of the evaluated set; the threshold stays fixed.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
CI_METRICS = ("recall", "specificity", "fpr", "precision", "f1", "roc_auc", "pr_auc")


def bootstrap_settings(n=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """How the `ci95` entries were computed, for the results files."""
    return {"method": "percentile bootstrap, stratified by class", "level": 0.95, "n": n, "seed": seed,
            "threshold": "fixed (the evaluated threshold)", "metrics": list(CI_METRICS)}


def bootstrap_ci(y_true, y_score, threshold, n=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """95 % percentile intervals {metric: [low, high]} for CI_METRICS at a fixed `threshold`.

    Lenses and non-lenses are resampled separately (with replacement, each to its own count). The metrics are defined
    as in binary_metrics. Metrics that need both classes (precision, F1, ROC-AUC, PR-AUC; recall needs lenses,
    specificity and FPR need non-lenses) are None when the evaluated set lacks that class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    if not len(y_true) or n <= 0:
        return {k: None for k in CI_METRICS}
    rng = np.random.default_rng(seed)
    y_pred = y_score >= threshold
    y_boot = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    values = {k: np.empty(n) for k in CI_METRICS}
    for b in range(n):
        i_pos, i_neg = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        tp, fp = int(y_pred[i_pos].sum()), int(y_pred[i_neg].sum())
        fn, tn = len(pos) - tp, len(neg) - fp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        values["recall"][b] = recall
        values["precision"][b] = precision
        values["specificity"][b] = tn / (tn + fp) if tn + fp else 0.0
        values["fpr"][b] = fp / (fp + tn) if fp + tn else 0.0
        values["f1"][b] = 2 * precision * recall / (precision + recall + 1e-8)
        if len(pos) and len(neg):
            s_boot = np.r_[y_score[i_pos], y_score[i_neg]]
            values["roc_auc"][b] = roc_auc_score(y_boot, s_boot)
            values["pr_auc"][b] = average_precision_score(y_boot, s_boot)
    defined = {"recall": len(pos) > 0, "specificity": len(neg) > 0, "fpr": len(neg) > 0}
    both = len(pos) > 0 and len(neg) > 0
    return {k: ([float(q) for q in np.percentile(v, [2.5, 97.5])] if defined.get(k, both) else None)
            for k, v in values.items()}


def save_curves(output_dir, y_true, y_score, prefix=""):
    """Write the ROC and PR curve points to `<prefix>roc_curve.csv` and `<prefix>pr_curve.csv` (C15).

    roc_curve.csv: fpr, tpr, threshold (the first row, threshold inf, is the (0, 0) corner).
    pr_curve.csv: recall, precision, threshold (the last row, recall 0 and precision 1, has no threshold).
    A predicted lens is lens_probability >= threshold, as everywhere else. Nothing is written for a one-class set.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(set(y_true.tolist())) < 2:
        return None
    fpr, tpr, roc_t = roc_curve(y_true, y_score)
    precision, recall, pr_t = precision_recall_curve(y_true, y_score)
    paths = Path(output_dir) / f"{prefix}roc_curve.csv", Path(output_dir) / f"{prefix}pr_curve.csv"
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_t}).to_csv(paths[0], index=False)
    pd.DataFrame({"recall": recall, "precision": precision,
                  "threshold": np.r_[pr_t, np.nan]}).to_csv(paths[1], index=False)
    return paths


def format_ci(ci, key):
    """' [low, high]' for log lines, or '' when there is no interval."""
    return f" [{ci[key][0]:.4f}, {ci[key][1]:.4f}]" if ci and ci.get(key) else ""


# How the decision threshold is chosen (on validation only). Stored in best_model.pth and test_results.json so that
# pipelines/evaluate_real.py applies the same frozen threshold to the real images. Change it together with
# best_f2_threshold once decision D3 (TODO.md) is settled.
THRESHOLD_RULE = "max_f2_on_validation"
# How the checkpoint is chosen (C13): the epoch with the lowest validation loss; early stopping uses the same quantity.
SELECTION_RULE = "min_val_loss"


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


def split_sizes(splits):
    """{split: {'n', 'lenses', 'nonlenses', 'sources': {source: n}}} for the results files."""
    out = {}
    for name, (paths, labels, sources) in splits.items():
        out[name] = {"n": len(paths), "lenses": int(sum(labels)), "nonlenses": int(len(labels) - sum(labels)),
                     "sources": {k: int(v) for k, v in pd.Series(sources, dtype=object).value_counts().items()}}
    return out


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------
def run_finetuning(args, model_name, model, model_config, head_params, train_transform, logger, extra_info=None):
    """Train `model` on the base dataset, keep the lowest-validation-loss epoch, evaluate on test, save everything.

    Args:
        model_name: 'lensnet' | 'deeplens' | 'efficientnet' | 'teglie' | 'gravit' (used by pipelines/load.py to
            rebuild the model).
        model: the model, freshly initialised (from-scratch runs), or pretrained with frozen parameters already set
            (requires_grad=False).
        model_config: kwargs that rebuild the architecture (stored in the checkpoint).
        head_params: parameters of the classification head (get head_learning_rate).
        train_transform: ModelInputTransform for training; evaluation uses it without augmentation.
    """
    output_dir = Path(args.output_dir)
    device = select_device(args.gpu, logger)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    eval_transform = train_transform.without_augmentation()
    data_info = {"train_data": resolve_train_data(args, logger), "split_seed": args.split_seed}
    recipe = getattr(args, "recipe", "thesis")
    logger.info(f"Training recipe '{recipe}'" + (" (pretrained side experiment)" if recipe == "finetune" else "") +
                f": lr {args.learning_rate:g}, weight decay {args.weight_decay:g}, batch {args.batch_size}, "
                f"max {args.epochs} epochs, warm-up {args.warmup_epochs}, augment {args.augment}, "
                f"selection on {SELECTION_RULE}, patience {args.early_stop_patience}")

    with open(output_dir / "system_info.json", "w") as f:
        json.dump({"model_name": model_name, "model_config": model_config, "recipe": recipe,
                   "selection": SELECTION_RULE, "input_preset": args.input_preset,
                   "transform": eval_transform.config(), **data_info, "arguments": vars(args), "python_version": sys.version, "pytorch_version": torch.__version__,
                   "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                   "device": device.type, "mixed_precision_used": use_amp, "hostname": platform.node(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   **(extra_info or {})}, f, indent=2, default=str)

    logger.info(f"Collecting files (--train_data {args.train_data})...")
    samples, quality_report = collect_from_args(args, logger)  # corrupted files dropped (datasets/quality.py)
    data_info["dropped_corrupted"] = dropped_counts(quality_report)
    logger.info(f"Data quality report saved to {save_report(quality_report, output_dir)}")
    splits = split_samples(samples, args.val_region, args.test_region, args.split_seed, logger)
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

    criterion = build_criterion(splits["train"][1], args, device, logger)
    scaler = make_grad_scaler(use_amp)

    def checkpoint(epoch, metrics):
        return {"model_name": model_name, "model_config": model_config, "transform": eval_transform.config(),
                "model_state_dict": unwrap(model).state_dict(), "epoch": epoch, "val_metrics": metrics,
                "args": vars(args)}

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_recall", "val_precision",
                               "val_specificity", "val_f1", "val_f2", "val_pr_auc", "val_roc_auc", "lr")}
    best_loss, epochs_no_improve = math.inf, 0
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
        if math.isfinite(val_loss) and val_loss < best_loss:
            best_loss, epochs_no_improve = val_loss, 0
            torch.save(checkpoint(epoch, {**m, "loss": val_loss}), output_dir / "best_model.pth")
            logger.info(f"  -> new best model saved (val loss {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience and epoch >= args.min_epochs:
                logger.info(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_path = output_dir / "best_model.pth"
    if not best_path.exists():
        logger.warning("No epoch had a finite validation loss; using the last epoch as best model")
        best_path = output_dir / "last_model.pth"
        torch.save(torch.load(best_path, map_location="cpu"), output_dir / "best_model.pth")
    ckpt = torch.load(best_path, map_location="cpu")
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Evaluating best model (epoch {ckpt['epoch']})")

    _, y_val, s_val = predict(model, val_loader, criterion, device, use_amp)
    threshold = best_f2_threshold(y_val, s_val)
    ckpt.update(threshold=threshold, threshold_rule=THRESHOLD_RULE)  # frozen for the real-data evaluation
    torch.save(ckpt, output_dir / "best_model.pth")
    _, y_test, s_test = predict(model, test_loader, criterion, device, use_amp)
    results = {**data_info, "recipe": recipe, "split_sizes": split_sizes(splits),
               "best_epoch": int(ckpt["epoch"]), "selection": SELECTION_RULE,
               "threshold_selected_on": "validation", "threshold_rule": THRESHOLD_RULE,
               "validation": binary_metrics(y_val, s_val, threshold),
               "test": {**binary_metrics(y_test, s_test, threshold), "ci95": bootstrap_ci(y_test, s_test, threshold)},
               "test_at_0.5": binary_metrics(y_test, s_test, 0.5),
               "bootstrap": bootstrap_settings()}
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    save_predictions(output_dir / "validation_predictions.csv", splits["val"][0], splits["val"][2], y_val, s_val, threshold)
    df = save_predictions(output_dir / "test_predictions.csv", splits["test"][0], splits["test"][2], y_test, s_test, threshold)
    curves = save_curves(output_dir, y_test, s_test)
    if curves:
        logger.info(f"Test ROC and PR curve points saved to {curves[0].name} and {curves[1].name}")

    t = results["test"]
    logger.info("=" * 80)
    logger.info(f"TEST RESULTS (threshold {threshold:.2f}, tuned for F2 on validation; "
                f"[95 % stratified bootstrap interval, {BOOTSTRAP_SAMPLES} resamples])")
    for key in ("recall", "specificity", "fpr", "precision", "f1", "f2", "pr_auc", "roc_auc", "acc", "prevalence"):
        logger.info(f"  {key:12s}: {t[key]:.4f}{format_ci(t['ci95'], key)}")
    logger.info(f"  confusion   : {t['confusion_matrix']}")
    for source, g in df.groupby("data_source"):
        logger.info(f"  {source}: {len(g)} samples, accuracy {100 * g['correct'].mean():.2f}%")
    logger.info("=" * 80)
    logger.info(f"Best model saved to {output_dir / 'best_model.pth'}")
    return results
