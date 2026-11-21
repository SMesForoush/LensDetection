"""External evaluation of a trained model on every real COSMOS-Web DR1 cutout (TODO C8).

    python -m pipelines.evaluate_real results/deeplens_prism --gpu
    python -m pipelines.evaluate_real results/deeplens_prism/best_model.pth \
        --cosmos_root cosmos_data --real_lens_dir mock-lens-modeling/real_cowls_lenses

The model, its input transform and its decision threshold are all frozen: they are read from the checkpoint
(pipelines/load.py). The threshold is the one the training run chose on its own validation set; there is
deliberately no option to set it, and this script reads no validation (or other PRISM) data. Every real image in
the manifest is scored, unsplit and unsubsampled (datasets.sources.all_real_samples).

Writes, next to the checkpoint unless --output_dir is given:
    real_predictions.csv   one row per real image, same columns as the training run's test_predictions.csv
    real_results.json      metrics at the frozen threshold, plus the threshold-free ROC-AUC and PR-AUC:
        overall            all real images
        per_source         real_lens_confident, real_lens_calibration (recall), lookalike, real_nonlens (specificity, FPR)
        per_tier           confident lenses + all non-lenses, calibration lenses + all non-lenses
    Every block with both classes has `prevalence` (lens fraction) and `pr_auc_random` (= prevalence, the PR-AUC of a
    random classifier) next to its PR-AUC.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.sources import REAL_SOURCES, add_real_data_arguments, all_real_samples  # noqa: E402
from pipelines.common import binary_metrics, make_loader, predict, save_predictions, select_device  # noqa: E402
from pipelines.load import frozen_threshold, load_from_checkpoint, read_checkpoint  # noqa: E402

LENS_SOURCES = ("real_lens_confident", "real_lens_calibration")
NONLENS_SOURCES = tuple(s for s in REAL_SOURCES if s not in LENS_SOURCES)
TIERS = {"confident": "real_lens_confident", "calibration": "real_lens_calibration"}


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("run", help="Run directory (uses its best_model.pth) or a checkpoint file")
    p.add_argument("--output_dir", help="Where to write real_predictions.csv and real_results.json "
                                        "(default: the checkpoint's directory)")
    add_real_data_arguments(p.add_argument_group("real data"))
    s = p.add_argument_group("system")
    s.add_argument("--batch_size", type=int, default=64)
    s.add_argument("--num_workers", type=int, default=8)
    s.add_argument("--prefetch_factor", type=int, default=4)
    s.add_argument("--gpu", action="store_true", help="Use a GPU: NVIDIA (CUDA) if present, else Apple Silicon (MPS)")
    return p.parse_args(argv)


def setup_logging(output_dir):
    logger = logging.getLogger("evaluate_real")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(Path(output_dir) / "evaluate_real.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def resolve_checkpoint(run):
    path = Path(run)
    if path.is_dir():
        path = path / "best_model.pth"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return path


def with_baseline(m, y_true):
    """binary_metrics plus FPR, prevalence and the PR-AUC of a random classifier (= prevalence)."""
    cm = m["confusion_matrix"]
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    return {**m, "n": int(len(y_true)), "lenses": int(np.sum(y_true)),
            "fpr": cm["FP"] / (cm["FP"] + cm["TN"]) if cm["FP"] + cm["TN"] else 0.0,
            "prevalence": prevalence, "pr_auc_random": prevalence}


def source_metrics(y_true, y_score, threshold):
    """One-class metrics for a single source: recall for lens sources, specificity and FPR for non-lens sources."""
    y_pred = y_score >= threshold
    out = {"n": int(len(y_true)), "label": int(y_true[0]) if len(y_true) else None,
           "mean_lens_probability": float(np.mean(y_score)) if len(y_score) else None,
           "median_lens_probability": float(np.median(y_score)) if len(y_score) else None}
    if out["label"] == 1:
        tp = int(np.sum(y_pred))
        out.update(TP=tp, FN=out["n"] - tp, recall=tp / out["n"])
    elif out["label"] == 0:
        fp = int(np.sum(y_pred))
        out.update(TN=out["n"] - fp, FP=fp, specificity=1 - fp / out["n"], fpr=fp / out["n"])
    return out


def real_metrics(y_true, y_score, sources, threshold):
    sources = np.asarray(sources)
    if set(sources) - set(REAL_SOURCES):
        raise ValueError(f"Unexpected sources {sorted(set(sources) - set(REAL_SOURCES))}")
    nonlens = np.isin(sources, NONLENS_SOURCES)
    results = {
        "overall": with_baseline(binary_metrics(y_true, y_score, threshold), y_true),
        "per_source": {s: source_metrics(y_true[sources == s], y_score[sources == s], threshold)
                       for s in REAL_SOURCES},
        "per_tier": {},
    }
    for tier, lens_source in TIERS.items():
        keep = nonlens | (sources == lens_source)
        results["per_tier"][tier] = {"sources": [lens_source, *NONLENS_SOURCES],
                                     **with_baseline(binary_metrics(y_true[keep], y_score[keep], threshold), y_true[keep])}
    return results


def main(argv=None):
    args = parse_arguments(argv)
    ckpt_path = resolve_checkpoint(args.run)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    ckpt = read_checkpoint(ckpt_path)
    threshold, rule, threshold_from = frozen_threshold(ckpt, ckpt_path)
    model, transform = load_from_checkpoint(ckpt)
    train_args = ckpt.get("args", {})
    # Runs from before --train_data existed trained on real data (the old default) unless they had no real data at all.
    train_data = train_args.get("train_data") or ("both" if train_args.get("include_prism") else "real")
    logger.info(f"Checkpoint {ckpt_path}: model '{ckpt['model_name']}', epoch {ckpt.get('epoch')}, "
                f"trained with --train_data {train_data}")
    logger.info(f"Frozen threshold {threshold:.4f} ({rule}), read from {threshold_from}")
    logger.info(f"Input transform: {transform.config()}")
    if train_data != "prism":
        logger.warning(f"This model was trained with --train_data {train_data}: real images were in its training, "
                       f"validation and test splits, so this is NOT an external evaluation")

    samples = all_real_samples(args, logger)
    paths = [s.path for s in samples]
    labels = [s.label for s in samples]
    sources = [s.source for s in samples]

    device = select_device(args.gpu, logger)
    model = model.to(device)
    loader = make_loader(paths, labels, transform, args, shuffle=False)
    loss, y_true, y_score = predict(model, loader, nn.CrossEntropyLoss(), device, use_amp=device.type == "cuda")
    if len(y_true) != len(samples) or not np.array_equal(y_true, np.asarray(labels)):
        raise RuntimeError(f"Scored {len(y_true)} of {len(samples)} real images, or out of order")

    results = {
        "checkpoint": str(ckpt_path), "model_name": ckpt["model_name"], "epoch": ckpt.get("epoch"),
        "train_data": train_data, "split_seed": train_args.get("split_seed"), "seed": train_args.get("seed"),
        "external_evaluation": train_data == "prism",
        "threshold": threshold, "threshold_rule": rule, "threshold_read_from": threshold_from,
        "transform": transform.config(), "real_manifest": str(args.real_manifest),
        "counts": {"n": len(samples), "lenses": int(sum(labels)), "nonlenses": len(labels) - int(sum(labels)),
                   "sources": {s: sources.count(s) for s in REAL_SOURCES}},
        "loss": loss,
        **real_metrics(y_true, y_score, sources, threshold),
    }
    with open(output_dir / "real_results.json", "w") as f:
        json.dump(results, f, indent=2)
    save_predictions(output_dir / "real_predictions.csv", paths, sources, y_true, y_score, threshold)

    o = results["overall"]
    logger.info("=" * 80)
    logger.info(f"REAL DATA ({o['n']} images, {o['lenses']} lenses, prevalence {o['prevalence']:.4f}), "
                f"threshold {threshold:.4f}")
    for key in ("recall", "specificity", "fpr", "precision", "f1", "roc_auc", "pr_auc", "pr_auc_random"):
        logger.info(f"  {key:14s}: {o[key]:.4f}")
    logger.info(f"  confusion     : {o['confusion_matrix']}")
    for source, m in results["per_source"].items():
        if not m["n"]:
            logger.warning(f"  {source}: no images in the manifest")
            continue
        key = "recall" if m["label"] == 1 else "fpr"
        logger.info(f"  {source:22s}: {m['n']:5d} images, {key} {m[key]:.4f}")
    for tier, m in results["per_tier"].items():
        logger.info(f"  tier {tier:11s}: recall {m['recall']:.4f}, precision {m['precision']:.4f}, "
                    f"ROC-AUC {m['roc_auc']:.4f}, PR-AUC {m['pr_auc']:.4f} (random {m['pr_auc_random']:.4f})")
    logger.info("=" * 80)
    logger.info(f"Wrote {output_dir / 'real_results.json'} and {output_dir / 'real_predictions.csv'}")
    return results


if __name__ == "__main__":
    main()
