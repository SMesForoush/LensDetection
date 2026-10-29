#!/usr/bin/env python3
"""Check that the PyTorch ports in models/teglie.py and models/deeplens.py reproduce the original Keras models.

Builds the original Keras graphs (transcribed from the upstream repositories), loads the published .h5
weights with Keras' own loader, loads the same file with our h5py-based loader, and compares predictions
on random inputs in inference mode. Needs TensorFlow + tf-keras (Keras 2); the training pipelines do not.

    TF_USE_LEGACY_KERAS=1 python scripts/verify_keras_ports.py \
        --teglie_weights "weights/teglie/Lens Detector 15.h5" \
        --deeplens_weights weights/lexactum/ground_cmu_deeplens_250epochs.h5
"""
import argparse
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models import deeplens, teglie


def keras_teglie_lens15(tf, keras):
    """https://github.com/margres/TEGLIE/blob/main/teglie_scripts/models.py::lens15 (without compile/load)."""
    L = keras.layers
    d_model = 128
    reg = keras.regularizers.l2(0.01)
    inputs = L.Input(shape=(101, 101, 4))
    x = inputs
    spec = [(16, 4, False, False), (16, 3, True, False), (32, 3, False, False), (32, 3, True, False),
            (64, 3, False, False), (64, 3, True, True), (128, 3, False, True), (128, 3, False, True)]
    for i, (filters, k, pool, drop) in enumerate(spec, start=1):
        x = L.Conv2D(filters, (k, k), padding="valid", activation="elu", kernel_regularizer=reg, name=f"Conv{i}")(x)
        if pool:
            x = L.MaxPooling2D((2, 2))(x)
        x = L.BatchNormalization(axis=3)(x)
        if drop:
            x = L.Dropout(0.5)(x)
    x = L.Reshape((25, d_model))(x)
    x = x * np.float32(np.sqrt(d_model)) + teglie.teglie_positional_encoding(10000, d_model).numpy()[:, :25, :]

    def mha(name=None):
        return L.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.2,
                                    use_bias=True, kernel_regularizer=reg, name=name)

    x = L.Add()([mha()(x, x), mha()(x, x)])
    x = L.Add()([mha()(x, x), mha("Att")(x, x)])
    dense = L.Dense(2 * d_model, activation="elu", name="dense1")(x)
    dense = L.Dropout(0.5)(dense)
    dense = L.Dense(d_model, activation="elu", name="dense2")(dense)
    dense = L.Dropout(0.5)(dense)
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, dense]))
    x = L.GlobalAveragePooling1D()(x)
    for units, name in ((1024, "dense4"), (512, "dense5"), (128, "dense6")):
        x = L.Dropout(0.5)(L.Dense(units, activation="elu", name=name)(x))
    return keras.models.Model(inputs, L.Dense(1, activation="sigmoid")(x))


def keras_cmu_deeplens(keras, image_size=101, channels=4):
    """https://github.com/DanielMagro97/LEXACTUM/blob/main/neural_networks/cmu_deeplens.py::cmu_deeplens."""
    L = keras.layers

    def block(x, f_in, f_out, down):
        if down:
            x = L.ELU()(L.BatchNormalization()(x))
            shortcut = L.Conv2D(f_out, (1, 1), strides=(2, 2), padding="same", activation="elu")(x)
            x = L.Conv2D(f_in, (1, 1), strides=(2, 2), padding="same", activation="elu")(x)
        else:
            shortcut = x
            x = L.ELU()(L.BatchNormalization()(x))
            x = L.Conv2D(f_in, (1, 1), padding="same", activation="elu")(x)
        x = L.ELU()(L.BatchNormalization()(x))
        x = L.Conv2D(f_in, (3, 3), padding="same", activation="elu")(x)
        x = L.ELU()(L.BatchNormalization()(x))
        x = L.Conv2D(f_out, (1, 1), padding="same", activation="elu")(x)
        return L.Add()([x, shortcut])

    inputs = L.Input((image_size, image_size, channels))
    x = L.BatchNormalization()(L.Conv2D(32, (7, 7), padding="same", activation="elu")(inputs))
    for f_in, f_out, down in deeplens.DEEPLENS_BLOCKS:
        x = block(x, f_in, f_out, down)
    x = L.Flatten()(L.AvgPool2D()(x))
    return keras.models.Model(inputs, L.Dense(1, activation="sigmoid")(x))


def compare(name, keras_model, torch_model, last_linear, x_nhwc, keras):
    """Compare the features entering the output layer (relative error) and the final probabilities.

    Comparing features as well as probabilities matters because a saturated sigmoid hides mismatches.
    """
    keras_feat_model = keras.models.Model(keras_model.input, keras_model.layers[-1].input)
    keras_feat = keras_feat_model.predict(x_nhwc, verbose=0)
    keras_p = keras_model.predict(x_nhwc, verbose=0).ravel()

    captured = {}
    hook = last_linear.register_forward_hook(lambda m, inp, out: captured.update(feat=inp[0]))
    torch_model.eval()
    with torch.no_grad():
        torch_p = torch.sigmoid(torch_model(torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2)))).numpy().ravel()
    hook.remove()
    torch_feat = captured["feat"].numpy()

    feat_err = np.abs(keras_feat - torch_feat).max() / (np.abs(keras_feat).max() + 1e-12)
    p_err = np.abs(keras_p - torch_p).max()
    print(f"[{name}] rel. feature err = {feat_err:.2e}, max |dp| = {p_err:.2e}, "
          f"keras p={np.round(keras_p[:4], 4)} torch p={np.round(torch_p[:4], 4)}")
    return max(feat_err, p_err)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teglie_weights", nargs="*", default=[])
    p.add_argument("--deeplens_weights", nargs="*", default=[])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    import tensorflow as tf
    import tf_keras as keras

    rng = np.random.default_rng(0)
    failures = 0
    for path in args.teglie_weights:
        km = keras_teglie_lens15(tf, keras)
        km.load_weights(path)  # Keras' own topological h5 loader
        tm = teglie.load_keras_h5(teglie.LensDetector15(), path)
        # TEGLIE inputs are clipped/scaled to [-100, 100]
        x = np.clip(rng.normal(0, 30, size=(args.n, 101, 101, 4)), -100, 100).astype(np.float32)
        failures += compare(f"TEGLIE {os.path.basename(path)}", km, tm, tm.head.out, x, keras) > args.tol
    for path in args.deeplens_weights:
        km = keras_cmu_deeplens(keras)
        km.load_weights(path)
        tm = deeplens.load_keras_h5(deeplens.CMUDeepLens(), path)
        # LEXACTUM inputs are per-band ZScale-normalised to [0, 1]
        x = rng.uniform(0, 1, size=(args.n, 101, 101, 4)).astype(np.float32)
        failures += compare(f"DeepLens {os.path.basename(path)}", km, tm, tm.fc, x, keras) > args.tol
    if failures:
        sys.exit(f"{failures} port(s) disagree with Keras beyond tol={args.tol}")
    print("All ports match the original Keras models.")


if __name__ == "__main__":
    main()
