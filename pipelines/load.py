"""Reload a model saved by any of the fine-tuning pipelines (best_model.pth / last_model.pth).

    from pipelines.load import load_finetuned, lens_probability
    model, transform = load_finetuned("results/teglie/best_model.pth")
    p = lens_probability(model, transform, raw_cutouts)   # raw_cutouts: (N, 4, 300, 300) base-dataset tensors

The checkpoint stores the architecture config and the exact input transform, so no pretrained weights
or network access are needed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform  # noqa: E402
from pipelines.common import two_class_logits  # noqa: E402


def build_architecture(model_name, model_config):
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


def load_finetuned(checkpoint_path, map_location="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model = build_architecture(ckpt["model_name"], ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ModelInputTransform(**ckpt["transform"])


@torch.no_grad()
def lens_probability(model, transform, images):
    """P(lens) for a batch of raw base-dataset images of shape (N, 4, 300, 300)."""
    device = next(model.parameters()).device
    x = torch.stack([transform(img) for img in images]).to(device)
    return torch.softmax(two_class_logits(model(x)), dim=1)[:, 1].cpu()
