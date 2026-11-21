"""Reload a model saved by any of the fine-tuning pipelines or by train.py (best_model.pth / last_model.pth).

    from pipelines.load import load_finetuned, lens_probability
    model, transform = load_finetuned("results/teglie/best_model.pth")
    p = lens_probability(model, transform, raw_cutouts)   # raw_cutouts: (N, 4, 300, 300) base-dataset tensors

The checkpoint stores the architecture config and the exact input transform, so no pretrained weights
or network access are needed. LensNet checkpoints written by train.py before it stored these are rebuilt from their
saved arguments (LensNet sees the raw base-dataset cutouts, so its transform is the identity).

`frozen_threshold` returns the decision threshold that the training run chose on its validation set.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform  # noqa: E402
from pipelines.common import two_class_logits  # noqa: E402


# train.py feeds LensNet the raw (4, 300, 300) base-dataset tensors; this transform config is the identity.
LENSNET_RAW_TRANSFORM = dict(bands=[0, 1, 2, 3], image_size=300, resize_mode="resize", scaling="none",
                             imagenet_norm=False)


def build_architecture(model_name, model_config):
    if model_name == "lensnet":
        from models.lensnet import LensNet
        return LensNet(**model_config)
    if model_name == "gravit":
        from models.gravit import build_gravit
        return build_gravit(**model_config)
    if model_name == "teglie":
        from models.teglie import LensDetector15
        return LensDetector15(**model_config)
    if model_name == "deeplens":
        from models.deeplens import CMUDeepLens
        return CMUDeepLens(**model_config)
    raise ValueError(f"Unknown model '{model_name}'")


def read_checkpoint(checkpoint_path, map_location="cpu"):
    """The checkpoint dict, with model_name / model_config / transform filled in for old train.py (LensNet) files."""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "model_name" not in ckpt:
        if "args" not in ckpt or "dropout" not in ckpt["args"]:
            raise ValueError(f"{checkpoint_path}: neither a pipelines/ nor a train.py checkpoint")
        ckpt.update(model_name="lensnet", model_config={"dropout": ckpt["args"]["dropout"], "in_channels": 4},
                    transform=dict(LENSNET_RAW_TRANSFORM))
    return ckpt


def load_from_checkpoint(ckpt):
    """(model in eval mode, its evaluation transform) from a dict returned by read_checkpoint."""
    model = build_architecture(ckpt["model_name"], ckpt["model_config"])
    state = {k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()}  # saved from nn.DataParallel
    model.load_state_dict(state)
    model.eval()
    return model, ModelInputTransform(**ckpt["transform"])


def load_finetuned(checkpoint_path, map_location="cpu"):
    return load_from_checkpoint(read_checkpoint(checkpoint_path, map_location))


def frozen_threshold(ckpt, checkpoint_path):
    """(threshold, rule, where it was read) for a checkpoint from read_checkpoint.

    The threshold was chosen on the run's validation set when training ended. It is read from the checkpoint (key
    `threshold`); runs older than that key have it only in the test_results.json next to the checkpoint, which is
    used if its best epoch matches the checkpoint's. There is deliberately no way to pass a threshold by hand."""
    if "threshold" in ckpt:
        return float(ckpt["threshold"]), ckpt.get("threshold_rule", "unknown"), "checkpoint"
    results_path = Path(checkpoint_path).with_name("test_results.json")
    if not results_path.is_file():
        raise ValueError(f"{checkpoint_path} has no stored threshold and there is no {results_path.name} next to it; "
                         f"re-run the training so that the validation threshold is saved in the checkpoint")
    results = json.loads(results_path.read_text())
    if results.get("threshold_selected_on") != "validation":
        raise ValueError(f"{results_path}: the threshold was not chosen on the validation set")
    best_epoch = results.get("best_epoch")
    if best_epoch is not None and int(best_epoch) != int(ckpt.get("epoch", -1)):
        raise ValueError(f"{results_path} belongs to epoch {best_epoch}, but {checkpoint_path} is epoch {ckpt.get('epoch')}")
    if "validation" in results:  # pipelines/common.py::run_finetuning
        threshold = results["validation"]["threshold"]
    elif "optimal_threshold" in results:  # train.py
        threshold = results["optimal_threshold"]
    else:
        raise ValueError(f"{results_path} holds no threshold")
    return float(threshold), results.get("threshold_rule", "max_f2_on_validation"), str(results_path)


@torch.no_grad()
def lens_probability(model, transform, images):
    """P(lens) for a batch of raw base-dataset images of shape (N, 4, 300, 300)."""
    device = next(model.parameters()).device
    x = torch.stack([transform(img) for img in images]).to(device)
    return torch.softmax(two_class_logits(model(x)), dim=1)[:, 1].cpu()
