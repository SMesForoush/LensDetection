"""PyTorch port of the TEGLIE / Lens Detector 15 model (Thuruthipilly et al. 2022; Grespan et al. 2024).

Original Keras code: https://github.com/margres/TEGLIE/blob/main/teglie_scripts/models.py (lens15)
Weights:             https://github.com/margres/TEGLIE/tree/main/TE_weights
    Lens_Detector_15.zip            -> "Lens Detector 15.h5"            (Bologna Lens Finding Challenge, before KiDS fine-tuning)
    Rotated_Flipped_Transposed.zip  -> "Rotated_Flipped_Transposed.h5"  (after fine-tuning on real KiDS data)

Architecture: 8-layer CNN backbone on 101x101x4 (band order r, i, g, u) -> 25 tokens of width 128 ->
sinusoidal positional encoding -> two pairs of parallel self-attention layers -> feed-forward with residual
+ LayerNorm -> global average pooling -> dense head. The port reproduces the Keras graph exactly,
including its non-standard positional encoding and Keras MultiHeadAttention parameterisation
(key_dim=128 *per head*), so the published .h5 weights load directly.

The model returns a single logit (the Keras sigmoid is dropped); use sigmoid(logit) for P(lens).
"""
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplens import _assign, load_keras_bn, load_keras_dense

KERAS_BN = dict(eps=1e-3, momentum=0.01)
D_MODEL = 128


def teglie_positional_encoding(position, d_model):
    """Exact copy of TEGLIE's preprocessing.positional_encoding (base 12800, sin/cos on alternate *positions*)."""
    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rads = pos * (1 / np.power(12800, (2 * (i // 2)) / np.float64(d_model)))
    angle_rads[0::2, :] = np.sin(angle_rads[0::2, :])
    angle_rads[1::2, :] = np.cos(angle_rads[1::2, :])
    return torch.tensor(angle_rads[np.newaxis], dtype=torch.float32)


class KerasMultiHeadAttention(nn.Module):
    """tf.keras.layers.MultiHeadAttention(num_heads, key_dim, value_dim, use_bias=True) for self-attention."""

    def __init__(self, embed_dim=D_MODEL, num_heads=8, key_dim=D_MODEL, value_dim=D_MODEL, dropout=0.2):
        super().__init__()
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.query = nn.Linear(embed_dim, num_heads * key_dim)
        self.key = nn.Linear(embed_dim, num_heads * key_dim)
        self.value = nn.Linear(embed_dim, num_heads * value_dim)
        self.output = nn.Linear(num_heads * value_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.query(x).view(b, t, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.num_heads, self.value_dim).transpose(1, 2)
        scores = torch.matmul(q / math.sqrt(self.key_dim), k.transpose(-1, -2))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.num_heads * self.value_dim)
        return self.output(out)


class LensDetector15(nn.Module):
    """TEGLIE Lens Detector 15.

    Args:
        in_channels: 4 (r, i, g, u) for the published weights.
        image_size: 101 for the published weights (gives a 5x5 = 25 token grid). Other sizes work
            with the same weights because the encoder is token-count agnostic.
        dropout: dropout rate used throughout the backbone and head (0.5 in the original).
        attention_dropout: attention-probability dropout (0.2 in the original).
    """

    def __init__(self, in_channels=4, image_size=101, dropout=0.5, attention_dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size

        def conv(cin, cout, k):
            return nn.Conv2d(cin, cout, k)  # padding='valid', stride 1

        self.backbone = nn.ModuleDict(dict(
            conv1=conv(in_channels, 16, 4), bn1=nn.BatchNorm2d(16, **KERAS_BN),
            conv2=conv(16, 16, 3), bn2=nn.BatchNorm2d(16, **KERAS_BN),
            conv3=conv(16, 32, 3), bn3=nn.BatchNorm2d(32, **KERAS_BN),
            conv4=conv(32, 32, 3), bn4=nn.BatchNorm2d(32, **KERAS_BN),
            conv5=conv(32, 64, 3), bn5=nn.BatchNorm2d(64, **KERAS_BN),
            conv6=conv(64, 64, 3), bn6=nn.BatchNorm2d(64, **KERAS_BN),
            conv7=conv(64, 128, 3), bn7=nn.BatchNorm2d(128, **KERAS_BN),
            conv8=conv(128, D_MODEL, 3), bn8=nn.BatchNorm2d(D_MODEL, **KERAS_BN),
        ))
        self.register_buffer("pos_encoding", teglie_positional_encoding(10000, D_MODEL), persistent=False)
        self.encoder = nn.ModuleDict(dict(
            att1=KerasMultiHeadAttention(dropout=attention_dropout),
            att2=KerasMultiHeadAttention(dropout=attention_dropout),
            att3=KerasMultiHeadAttention(dropout=attention_dropout),
            att4=KerasMultiHeadAttention(dropout=attention_dropout),
            dense1=nn.Linear(D_MODEL, 2 * D_MODEL),
            dense2=nn.Linear(2 * D_MODEL, D_MODEL),
            norm=nn.LayerNorm(D_MODEL, eps=1e-6),
        ))
        self.head = nn.ModuleDict(dict(
            dense4=nn.Linear(D_MODEL, 1024),
            dense5=nn.Linear(1024, 512),
            dense6=nn.Linear(512, 128),
            out=nn.Linear(128, 1),
        ))
        self.dropout = nn.Dropout(dropout)

    def stage_modules(self):
        return [self.backbone, self.encoder]

    def head_modules(self):
        return [self.head]

    def forward(self, x):
        b, d = self.backbone, self.dropout
        x = b.bn1(F.elu(b.conv1(x)))
        x = b.bn2(F.max_pool2d(F.elu(b.conv2(x)), 2))
        x = b.bn3(F.elu(b.conv3(x)))
        x = b.bn4(F.max_pool2d(F.elu(b.conv4(x)), 2))
        x = b.bn5(F.elu(b.conv5(x)))
        x = d(b.bn6(F.max_pool2d(F.elu(b.conv6(x)), 2)))
        x = d(b.bn7(F.elu(b.conv7(x))))
        x = d(b.bn8(F.elu(b.conv8(x))))

        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C) like tf.reshape on NHWC
        x = x * math.sqrt(D_MODEL) + self.pos_encoding[:, :x.shape[1], :]

        e = self.encoder
        x = e.att1(x) + e.att2(x)
        x = e.att3(x) + e.att4(x)
        ff = d(F.elu(e.dense2(d(F.elu(e.dense1(x))))))
        x = e.norm(x + ff).mean(dim=1)

        h = self.head
        x = d(F.elu(h.dense4(x)))
        x = d(F.elu(h.dense5(x)))
        x = d(F.elu(h.dense6(x)))
        return h.out(x)


# --------------------------------------------------------------------------------------
# Keras .h5 weight loading (h5py only - TensorFlow is not required)
# --------------------------------------------------------------------------------------
def _read_keras_h5(h5_path):
    """Return [(layer_name, {weight_path: array})] in the file's layer order."""
    import h5py

    layers = []
    with h5py.File(h5_path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        for name in g.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            weight_names = [w.decode() if isinstance(w, bytes) else w for w in g[name].attrs["weight_names"]]
            if weight_names:
                layers.append((name, {w: np.array(g[name][w]) for w in weight_names}))
    return layers


def _short(weights):
    """{'Conv1/kernel:0': a} -> {'kernel': a}; MHA weights keep their sub-layer: {'query/kernel': a}."""
    out = {}
    for path, arr in weights.items():
        parts = path.split(":")[0].split("/")
        out["/".join(parts[-2:]) if parts[-2] in ("query", "key", "value", "attention_output") else parts[-1]] = arr
    return out


def _ordered(layers, prefix):
    """Weights of auto-named layers `prefix`, `prefix_1`, ... sorted by Keras creation order."""
    found = []
    for name, w in layers:
        m = re.fullmatch(re.escape(prefix) + r"(?:_(\d+))?", name)
        if m:
            found.append((int(m.group(1) or 0), w))
    return [w for _, w in sorted(found, key=lambda t: t[0])]


def _load_keras_mha(mha, w):
    for torch_name, keras_name in (("query", "query"), ("key", "key"), ("value", "value")):
        lin = getattr(mha, torch_name)
        kernel = w[f"{keras_name}/kernel"]  # (embed, heads, dim)
        _assign(lin.weight, kernel.reshape(kernel.shape[0], -1).T)
        _assign(lin.bias, w[f"{keras_name}/bias"].reshape(-1))
    kernel = w["attention_output/kernel"]  # (heads, dim, embed)
    _assign(mha.output.weight, kernel.reshape(-1, kernel.shape[-1]).T)
    _assign(mha.output.bias, w["attention_output/bias"])


def load_keras_h5(model, h5_path, skip_head=False):
    """Load TEGLIE Lens Detector 15 Keras weights into `model` (in place)."""
    layers = [(name, _short(w)) for name, w in _read_keras_h5(h5_path)]
    by_name = dict(layers)
    b, e, h = model.backbone, model.encoder, model.head

    for i in range(1, 9):
        w = by_name[f"Conv{i}"]
        _assign(b[f"conv{i}"].weight, w["kernel"].transpose(3, 2, 0, 1))
        _assign(b[f"conv{i}"].bias, w["bias"])

    # Auto-named layers: the numeric suffix gives Keras creation order. The 4th attention layer is
    # named 'Att' in the fine-tuned weights but auto-named in the original ones.
    bns = _ordered(layers, "batch_normalization")
    mhas = _ordered(layers, "multi_head_attention") + ([by_name["Att"]] if "Att" in by_name else [])
    if len(bns) != 8 or len(mhas) != 4:
        raise ValueError(f"Expected 8 BN and 4 MHA layers, found {len(bns)} and {len(mhas)} in {h5_path}")
    for i, w in enumerate(bns, start=1):
        load_keras_bn(b[f"bn{i}"], w)
    for i, w in enumerate(mhas, start=1):
        _load_keras_mha(e[f"att{i}"], w)

    load_keras_dense(e.dense1, by_name["dense1"])
    load_keras_dense(e.dense2, by_name["dense2"])
    ln = _ordered(layers, "layer_normalization")[0]
    _assign(e.norm.weight, ln["gamma"])
    _assign(e.norm.bias, ln["beta"])

    if not skip_head:
        for name in ("dense4", "dense5", "dense6"):
            load_keras_dense(h[name], by_name[name])
        out = _ordered(layers, "dense")
        if len(out) != 1:
            raise ValueError(f"Could not identify the output dense layer in {h5_path}")
        load_keras_dense(h.out, out[0])
    return model
