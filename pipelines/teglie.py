#!/usr/bin/env python3
"""Train the TEGLIE / Lens Detector 15 CNN+transformer (Thuruthipilly et al. 2022; Grespan et al. 2024) from scratch,
or fine-tune its published weights.

    python -m pipelines.teglie --output_dir ... --gpu [--train_data prism|real|both --prism_dir ...]
    python -m pipelines.teglie --weights "weights/teglie/Lens Detector 15.h5" --input_preset native \
        --output_dir ... --gpu

Without --weights (the thesis experiment) the network starts from PyTorch's default initialisation and every layer
is trained; --freeze_stages needs --weights.

Weights (see scripts/download_weights.sh):
    "Lens Detector 15.h5"            trained on the Bologna Lens Finding Challenge (KiDS-like) simulations
    Rotated_Flipped_Transposed.h5    the same model after fine-tuning on real KiDS data (Grespan et al. 2024)

Inputs: by default (--input_preset thesis) the shared thesis transform, 4 bands 0,1,2,3 at 300x300 with a per-band
asinh stretch (datasets/transforms.py::THESIS_TRANSFORM). --input_preset native restores TEGLIE's own input: 101x101
cutouts with 4 bands in the order r, i, g, u (--bands, default 2,3,1,0), clipped and scaled to [-100, 100].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import parse_bands  # noqa: E402
from models.teglie import LensDetector15, load_keras_h5  # noqa: E402
from pipelines.common import (build_arg_parser, build_input_transform, freeze_modules, run_finetuning,  # noqa: E402
                              set_random_seeds, setup_logging)

NATIVE_BANDS, NATIVE_SIZE = "2,3,1,0", 101  # TEGLIE: r, i, g, u slots at 101x101


def build_model(cfg):
    return LensDetector15(**cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("TEGLIE")
    g.add_argument("--weights", help="Keras .h5 weights to fine-tune ('Lens Detector 15.h5' or "
                                     "'Rotated_Flipped_Transposed.h5'). Default: none, train from random initialisation")
    g.add_argument("--image_size", type=int,
                   help=f"Model input size, --input_preset native only (default {NATIVE_SIZE}, the size of the weights)")
    g.add_argument("--freeze_stages", type=int, default=0, choices=[0, 1, 2],
                   help="0 = train everything, 1 = freeze CNN backbone, 2 = freeze backbone + transformer encoder; "
                        "needs --weights")
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--attention_dropout", type=float, default=0.2)
    g.add_argument("--scaling_mode", choices=["robust", "absolute"], default="robust",
                   help="--input_preset native only. robust: clip each band at its --clip_pct percentile; absolute: "
                        "original TEGLIE clip at +-1e-9 after multiplying by --flux_scale")
    g.add_argument("--clip_pct", type=float, default=99.5, help="--input_preset native only")
    g.add_argument("--flux_scale", type=float, default=1.0, help="--input_preset native only")
    # Original training: Adam, lr 1e-4 with exponential decay, binary cross-entropy, l2(0.01) on conv/attention
    p.set_defaults(learning_rate=1e-4, weight_decay=1e-2, batch_size=64, epochs=50, min_epochs=20)
    args = p.parse_args(argv)
    if args.freeze_stages and not args.weights:
        p.error("--freeze_stages needs --weights: freezing randomly initialised layers is never wanted")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "finetune_teglie")

    bands = parse_bands(args.bands or NATIVE_BANDS)
    if len(bands) != 4:
        raise ValueError(f"TEGLIE expects 4 input bands (r, i, g, u), got --bands {args.bands}")
    scaling = dict(mode=args.scaling_mode, clip_pct=args.clip_pct) if args.scaling_mode == "robust" \
        else dict(mode="absolute", flux_scale=args.flux_scale)
    transform = build_input_transform(args, dict(bands=bands, image_size=args.image_size or NATIVE_SIZE,
                                                 resize_mode=args.resize_mode, scaling="teglie", **scaling), logger)
    model_config = dict(in_channels=len(transform.bands), image_size=transform.image_size, dropout=args.dropout,
                        attention_dropout=args.attention_dropout)
    model = build_model(model_config)
    if args.weights:
        load_keras_h5(model, args.weights)
        freeze_modules(model.stage_modules()[:args.freeze_stages])
        logger.info(f"Loaded TEGLIE weights from {args.weights}; frozen stages: {args.freeze_stages}")
        if args.input_preset != "native":
            logger.warning("Pretrained weights with the thesis input preset: the weights expect --input_preset native")
    else:
        logger.info("No --weights: training TEGLIE from random initialisation")

    head_params = [p for m in model.head_modules() for p in m.parameters()]
    run_finetuning(args, "teglie", model, model_config, head_params, transform, logger,
                   extra_info={"pretrained": str(args.weights) if args.weights else None})


if __name__ == "__main__":
    main()
