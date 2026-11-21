#!/usr/bin/env python3
"""Grad-CAM maps of the lens score for images listed in a predictions CSV.

    python scripts/gradcam.py results/lensnet_prism                                  # test_predictions.csv
    python scripts/gradcam.py results/lensnet_prism --predictions real_predictions.csv --outcomes FP,FN \
        --cosmos_root cosmos_data --num_samples 24

The checkpoint (a run directory or a .pth file) is reloaded with pipelines/load.py, so every model saved by the
pipelines works, and so do LensNet checkpoints of the legacy train.py. Images are read from the `file_path` column of
the predictions CSV (written by run_finetuning or pipelines/evaluate_real.py), go through the checkpoint's own
evaluation transform, and the map is computed for the lens logit at the output of `--layer` (default `pool5` for
LensNet; other models need `--layer`, a module name from model.named_modules()).

Writes `gradcam_<predictions stem>.png` (one panel per image: the mean of the model's input bands with the map on top,
titled with the true label, outcome and p(lens) from the CSV) and `gradcam_<predictions stem>.csv` (the rows shown),
next to the checkpoint unless --output_dir is given. Moved here from train.py (TODO C10) for the error analysis (E21).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.common import two_class_logits  # noqa: E402

# LensNet: pool5 is the last rectified feature map (9 x 9 at 300 px), the one global average pooling reads. conv5
# (what train.py used) is taken before BatchNorm and ReLU, so its Grad-CAM can be negative everywhere and come out blank.
DEFAULT_LAYERS = {"lensnet": "pool5"}
OUTCOMES = ("TP", "TN", "FP", "FN")


class GradCAM:
    """Gradient-weighted class activation mapping (Selvaraju et al. 2017) at the output of `target_layer`.

    Works for models with one sigmoid logit or two softmax logits (see two_class_logits); class 1 is the lens."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        self._handle.remove()

    def generate_cam(self, input_tensor, target_class=1):
        """(B, C, H, W) model input -> Grad-CAM map(s) in [0, 1] at the input resolution, squeezed, as numpy."""
        self.model.eval()
        self.gradients = None
        with torch.enable_grad():
            logits = two_class_logits(self.model(input_tensor))
            self.model.zero_grad()
            logits[:, target_class].sum().backward()
        if self.gradients is None:
            raise RuntimeError("no gradient reached the target layer (are the model parameters frozen?)")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy()


def resolve_checkpoint(path):
    path = Path(path)
    return path / "best_model.pth" if path.is_dir() else path


def pick_rows(df, outcomes, num_samples, seed):
    """Up to num_samples rows, split evenly over the requested outcomes, drawn at random with a fixed seed."""
    groups = [df[df["outcome"] == o] for o in outcomes]
    groups = [g for g in groups if len(g)]
    if not groups:
        return df.iloc[:0]
    per_group = max(1, num_samples // len(groups))
    picked = [g.sample(n=min(per_group, len(g)), random_state=seed) for g in groups]
    return pd.concat(picked)


def resolve_image_path(file_path, args):
    """CSV paths are absolute for PRISM; real images may be relative to --cosmos_root."""
    p = Path(file_path)
    if p.is_file() or p.is_absolute() or args.cosmos_root is None:
        return p
    return Path(args.cosmos_root) / p


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Run directory (uses its best_model.pth) or a .pth file")
    p.add_argument("--predictions", default="test_predictions.csv",
                   help="Predictions CSV; a bare file name is looked up next to the checkpoint")
    p.add_argument("--outcomes", default="TP,TN,FP,FN", help="Comma-separated outcomes to draw from")
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0, help="Seed for drawing the rows")
    p.add_argument("--layer", help="Target module name (default per model: " +
                   ", ".join(f"{k}={v}" for k, v in DEFAULT_LAYERS.items()) + ")")
    p.add_argument("--cosmos_root", help="Prefix for relative file paths in the CSV")
    p.add_argument("--output_dir", help="Default: the checkpoint's directory")
    p.add_argument("--gpu", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    from datasets.base import LensDatasetMemoryEfficient
    from pipelines.common import select_device
    from pipelines.load import load_from_checkpoint, read_checkpoint

    args = parse_args(argv)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    ckpt = read_checkpoint(ckpt_path)
    model_name = ckpt["model_name"]
    model, transform = load_from_checkpoint(ckpt)
    device = select_device(args.gpu)
    model.to(device)

    layer_name = args.layer or DEFAULT_LAYERS.get(model_name)
    if layer_name is None:
        raise SystemExit(f"No default Grad-CAM layer for '{model_name}'; pass --layer (see model.named_modules())")
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        raise SystemExit(f"'{model_name}' has no module named '{layer_name}'")
    for param in model.parameters():
        param.requires_grad_(True)

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        pred_path = ckpt_path.parent / args.predictions
    df = pd.read_csv(pred_path)
    outcomes = [o.strip().upper() for o in args.outcomes.split(",") if o.strip()]
    unknown = set(outcomes) - set(OUTCOMES)
    if unknown:
        raise SystemExit(f"Unknown outcomes {sorted(unknown)}; choose from {OUTCOMES}")
    rows = pick_rows(df, outcomes, args.num_samples, args.seed)
    if rows.empty:
        raise SystemExit(f"{pred_path} has no rows with outcome in {outcomes}")

    paths = [resolve_image_path(f, args) for f in rows["file_path"]]
    base = LensDatasetMemoryEfficient([str(p) for p in paths], rows["true_label"].tolist(), track_statistics=False)
    gradcam = GradCAM(model, layer)

    n_cols = 4
    n_rows = (len(rows) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.3 * n_rows), squeeze=False)
    heatmap = None
    for ax, (idx, row) in zip(axes.flat, enumerate(rows.itertuples())):
        x = transform(base[idx][0])[None].to(device)
        cam = gradcam.generate_cam(x, target_class=1)
        img = x[0].mean(dim=0).cpu().numpy()
        ax.imshow(img, cmap="gray", origin="lower")
        heatmap = ax.imshow(cam, cmap="jet", alpha=0.4, origin="lower", vmin=0, vmax=1)
        truth = "lens" if row.true_label == 1 else "non-lens"
        ax.set_title(f"{truth} | {row.outcome} | p(lens) {row.lens_probability:.2f}\n{Path(row.file_path).stem[:40]}",
                     fontsize=9, color="green" if row.outcome in ("TP", "TN") else "red")
        ax.axis("off")
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    gradcam.remove()
    fig.suptitle(f"Grad-CAM of the lens logit at '{layer_name}' ({model_name}, epoch {ckpt.get('epoch')})")
    fig.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.6, label="relative importance")

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"gradcam_{pred_path.stem}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rows.to_csv(out_dir / f"gradcam_{pred_path.stem}.csv", index=False)
    print(f"Grad-CAM of {len(rows)} images ({', '.join(outcomes)}) saved to {png}")


if __name__ == "__main__":
    main()
