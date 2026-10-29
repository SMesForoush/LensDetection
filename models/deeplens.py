"""PyTorch port of CMU DeepLens as re-implemented in LEXACTUM (Magro et al. 2021).

Original Keras code: https://github.com/DanielMagro97/LEXACTUM/blob/main/neural_networks/cmu_deeplens.py
Trained weights:     https://doi.org/10.5281/zenodo.4299924 (e.g. ground_cmu_deeplens_250epochs.h5)

The port mirrors the Keras graph layer for layer so the published .h5 weights load directly:
- every Conv2D carries its own ELU *and* is followed by a separate BN + ELU (as in the original)
- Keras BatchNormalization defaults (eps=1e-3, momentum=0.99 -> torch momentum=0.01)
- the dense head flattens in channels-last (H, W, C) order, like keras.layers.Flatten

The model returns a single logit (the Keras sigmoid is dropped); use sigmoid(logit) for P(lens).
"""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_BN = dict(eps=1e-3, momentum=0.01)

# (filters_in, filters_out, downsampling) for the 15 residual blocks
DEEPLENS_BLOCKS = (
    [(16, 32, False)] * 3
    + [(32, 64, True)] + [(32, 64, False)] * 2
    + [(64, 128, True)] + [(64, 128, False)] * 2
    + [(128, 256, True)] + [(128, 256, False)] * 2
    + [(256, 512, True)] + [(256, 512, False)] * 2
)


class ConvELU(nn.Conv2d):
    """Conv2D(..., activation='elu') with Keras 'same' padding (stride-2 convs here are all 1x1)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        return F.elu(super().forward(x))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, filters_in, filters_out, downsampling):
        super().__init__()
        self.downsampling = downsampling
        stride = 2 if downsampling else 1
        # Attribute order == Keras layer creation order (used by load_keras_h5)
        if downsampling:
            self.bn0 = nn.BatchNorm2d(in_channels, **KERAS_BN)
            self.shortcut = ConvELU(in_channels, filters_out, 1, stride=2)
            self.conv1 = ConvELU(in_channels, filters_in, 1, stride=stride)
        else:
            self.bn0 = nn.BatchNorm2d(in_channels, **KERAS_BN)
            self.conv1 = ConvELU(in_channels, filters_in, 1)
        self.bn1 = nn.BatchNorm2d(filters_in, **KERAS_BN)
        self.conv2 = ConvELU(filters_in, filters_in, 3)
        self.bn2 = nn.BatchNorm2d(filters_in, **KERAS_BN)
        self.conv3 = ConvELU(filters_in, filters_out, 1)

    def forward(self, x):
        if self.downsampling:
            x = F.elu(self.bn0(x))
            shortcut = self.shortcut(x)
            x = self.conv1(x)
        else:
            shortcut = x
            x = self.conv1(F.elu(self.bn0(x)))
        x = self.conv2(F.elu(self.bn1(x)))
        x = self.conv3(F.elu(self.bn2(x)))
        return x + shortcut


class CMUDeepLens(nn.Module):
    """CMU DeepLens (Lanusse et al. 2018) in the LEXACTUM layout.

    Args:
        in_channels: 4 for the ground-based weights (band order R, I, G, U), 1 for space-based.
        image_size: 101 for the published weights. Any other size changes the dense head shape,
            so only the convolutional weights can be transferred (see load_keras_h5(skip_head=True)).
        dropout: optional dropout before the dense head (the original has none).
    """

    def __init__(self, in_channels=4, image_size=101, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.stem = ConvELU(in_channels, 32, 7)
        self.stem_bn = nn.BatchNorm2d(32, **KERAS_BN)
        blocks, ch = [], 32
        for f_in, f_out, down in DEEPLENS_BLOCKS:
            blocks.append(ResNetBlock(ch, f_in, f_out, down))
            ch = f_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(dropout)
        size = image_size
        for _ in range(4):
            size = (size + 1) // 2
        size //= 2
        self.fc = nn.Linear(size * size * 512, 1)

    # Parameter groups used by the fine-tuning pipeline for freezing
    def stage_modules(self):
        """Stem followed by the five residual groups (3 blocks each)."""
        stages = [nn.ModuleList([self.stem, self.stem_bn])]
        for g in range(5):
            stages.append(nn.ModuleList(self.blocks[3 * g:3 * g + 3]))
        return stages

    def head_modules(self):
        return [self.fc]

    def forward(self, x):
        x = self.stem_bn(self.stem(x))
        x = self.blocks(x)
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).flatten(1)  # Keras Flatten is channels-last
        return self.fc(self.dropout(x))


# --------------------------------------------------------------------------------------
# Keras .h5 weight loading (h5py only - TensorFlow is not required)
# --------------------------------------------------------------------------------------
def _keras_layers_by_type(h5_path):
    """Return {'conv2d': [...], 'batch_normalization': [...], 'dense': [...]} in creation order."""
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        for name in g.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            weight_names = [w.decode() if isinstance(w, bytes) else w for w in g[name].attrs["weight_names"]]
            if not weight_names:
                continue
            m = re.match(r"^(conv2d|batch_normalization|dense)(?:_(\d+))?$", name)
            if m is None:
                raise ValueError(f"Unexpected layer '{name}' in {h5_path}")
            weights = {w.split("/")[-1].split(":")[0]: np.array(g[name][w]) for w in weight_names}
            out.setdefault(m.group(1), []).append((int(m.group(2) or 0), weights))
    return {k: [w for _, w in sorted(v, key=lambda t: t[0])] for k, v in out.items()}


def _assign(param, array):
    array = torch.from_numpy(np.ascontiguousarray(array))
    if tuple(param.shape) != tuple(array.shape):
        raise ValueError(f"Shape mismatch: model {tuple(param.shape)} vs keras {tuple(array.shape)}")
    with torch.no_grad():
        param.copy_(array)


def load_keras_conv(conv, w):
    _assign(conv.weight, w["kernel"].transpose(3, 2, 0, 1))
    _assign(conv.bias, w["bias"])


def load_keras_bn(bn, w):
    _assign(bn.weight, w["gamma"])
    _assign(bn.bias, w["beta"])
    _assign(bn.running_mean, w["moving_mean"])
    _assign(bn.running_var, w["moving_variance"])


def load_keras_dense(linear, w):
    _assign(linear.weight, w["kernel"].T)
    _assign(linear.bias, w["bias"])


def load_keras_h5(model, h5_path, skip_head=False):
    """Load LEXACTUM CMU DeepLens Keras weights into `model` (in place).

    skip_head=True keeps the dense layer randomly initialised, which is required when the
    model was built for an image size other than the one the weights were trained on.
    """
    layers = _keras_layers_by_type(h5_path)
    convs = [m for m in model.modules() if isinstance(m, ConvELU)]
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if len(convs) != len(layers["conv2d"]) or len(bns) != len(layers["batch_normalization"]):
        raise ValueError(f"Layer count mismatch: torch {len(convs)} conv / {len(bns)} bn, "
                         f"keras {len(layers['conv2d'])} conv / {len(layers['batch_normalization'])} bn")
    # Keras creates the BN of a downsampling block before its shortcut conv, and modules()
    # yields attributes in definition order, which matches Keras creation order per type.
    for conv, w in zip(convs, layers["conv2d"]):
        load_keras_conv(conv, w)
    for bn, w in zip(bns, layers["batch_normalization"]):
        load_keras_bn(bn, w)
    if not skip_head:
        load_keras_dense(model.fc, layers["dense"][0])
    return model
