"""GraViT lens-finding models (Parlange et al. 2025) from the HuggingFace Hub.

Repos are named parlange/<family>-gravit-<experiment>, e.g. parlange/vit-gravit-c3:
    experiment letter: a = C21, b = J24, c = C21+J24, s = C21 subsampled
    experiment digit:  1 = head only, 2 = half of the layers, 3 = all layers fine-tuned
https://github.com/parlange/gravit, https://huggingface.co/collections/parlange/gravit

The `architecture` field in several repos' config.json is wrong (pit, mlp-mixer and both twins variants
say vit_base_patch16_224; cvt says cvt_13_224, which timm does not have, while the weights are a
HuggingFace transformers CvT-21). So `timm.create_model('hf-hub:...')` only works for some families.
The architecture is therefore taken from GRAVIT_TIMM_ARCHS below; each one was checked to match the
published safetensors keys and shapes exactly.

All models take 3x224x224 ImageNet-normalised input and return 2 logits (index 1 = lens).
"""
import re

import torch
import torch.nn as nn

GRAVIT_TIMM_ARCHS = {
    "vit": "vit_base_patch16_224",
    "deit": "deit_base_patch16_224",
    "deit3": "deit3_base_patch16_224",
    "cait": "cait_s24_224",
    "swin": "swin_base_patch4_window7_224",
    "pit": "pit_b_224",
    "mlp-mixer": "mixer_b16_224",
    "twins_svt": "twins_svt_base",
    "twins_pcpvt": "twins_pcpvt_base",
}
GRAVIT_FAMILIES = tuple(GRAVIT_TIMM_ARCHS) + ("cvt",)


def gravit_family(repo_id):
    m = re.fullmatch(r"(?:.*/)?(.+)-gravit-[a-z]\d", repo_id)
    if m is None or m.group(1) not in GRAVIT_FAMILIES:
        raise ValueError(f"'{repo_id}' is not a GraViT repo id; expected parlange/<family>-gravit-<exp> "
                         f"with family in {GRAVIT_FAMILIES}")
    return m.group(1)


class HFCvtClassifier(nn.Module):
    """transformers CvtForImageClassification wrapped to return a logits tensor like a timm model."""

    def __init__(self, depth=(1, 4, 16), num_classes=2, drop_path_rate=0.0):
        super().__init__()
        from transformers import CvtConfig, CvtForImageClassification

        config = CvtConfig(depth=list(depth), num_labels=num_classes,
                           drop_path_rate=[0.0, 0.0, drop_path_rate])
        self.model = CvtForImageClassification(config)

    def get_classifier(self):
        return self.model.classifier

    def forward(self, x):
        return self.model(pixel_values=x).logits


def _cvt_depth(state_dict):
    depth = [0, 0, 0]
    for k in state_dict:
        m = re.match(r"cvt\.encoder\.stages\.(\d+)\.layers\.(\d+)\.", k)
        if m:
            s, i = int(m.group(1)), int(m.group(2))
            depth[s] = max(depth[s], i + 1)
    return tuple(depth)


def build_gravit(family, drop_path_rate=0.1, cvt_depth=(1, 4, 16)):
    """Build the (randomly initialised) architecture for a GraViT family."""
    if family == "cvt":
        return HFCvtClassifier(depth=cvt_depth, num_classes=2, drop_path_rate=drop_path_rate)
    import timm

    return timm.create_model(GRAVIT_TIMM_ARCHS[family], pretrained=False, num_classes=2,
                             drop_path_rate=drop_path_rate)


def load_gravit(repo_id="parlange/vit-gravit-c3", weights=None, drop_path_rate=0.1):
    """Build a GraViT model and load its published weights.

    Args:
        repo_id: HuggingFace repo id (also used to infer the architecture family).
        weights: optional local path to the repo's model.safetensors (for offline compute nodes).
            If omitted, the file is fetched with huggingface_hub (honours HF_HOME / HF_HUB_OFFLINE).
    Returns:
        (model, config) where config rebuilds the architecture via build_gravit(**config).
    """
    from safetensors.torch import load_file

    family = gravit_family(repo_id)
    if weights is None:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(repo_id, "model.safetensors")
    state_dict = load_file(weights)

    config = dict(family=family, drop_path_rate=drop_path_rate)
    if family == "cvt":
        config["cvt_depth"] = _cvt_depth(state_dict)
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    model = build_gravit(**config)
    model.load_state_dict(state_dict, strict=True)
    return model, config


def classifier_parameters(model):
    """Parameters of the classification head (trained in every fine-tuning mode)."""
    return list(model.get_classifier().parameters())


def set_gravit_trainable(model, mode):
    """GraViT fine-tuning strategies: 'head' (exp. 1), 'half' (exp. 2) or 'all' (exp. 3).

    'half' freezes the first half of the backbone's parameter tensors in network order (patch embedding
    and early blocks) and trains the rest plus the head.
    """
    head_ids = {id(p) for p in classifier_parameters(model)}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if mode == "all":
        n_frozen = 0
    elif mode == "half":
        n_frozen = len(backbone) // 2
    elif mode == "head":
        n_frozen = len(backbone)
    else:
        raise ValueError(f"Unknown GraViT fine-tuning mode '{mode}'")
    for i, p in enumerate(backbone):
        p.requires_grad_(i >= n_frozen)
    for p in classifier_parameters(model):
        p.requires_grad_(True)
    return model
