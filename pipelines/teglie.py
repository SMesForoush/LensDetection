#!/usr/bin/env python3
"""Fine-tune the TEGLIE / Lens Detector 15 CNN+transformer (Thuruthipilly et al. 2022; Grespan et al. 2024).

    python -m pipelines.teglie --weights "weights/teglie/Lens Detector 15.h5" \
        --mock_data_dir ... --real_lens_dir ... --real_nonlens_dir ... --output_dir ... --gpu

Weights (see scripts/download_weights.sh):
    "Lens Detector 15.h5"            trained on the Bologna Lens Finding Challenge (KiDS-like) simulations
    Rotated_Flipped_Transposed.h5    the same model after fine-tuning on real KiDS data (Grespan et al. 2024)

The model was trained on 101x101 cutouts with 4 bands in the order r, i, g, u, clipped and scaled to
[-100, 100]. --bands chooses which base-data channels fill those four slots.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform, parse_bands  # noqa: E402
from models.teglie import LensDetector15, load_keras_h5  # noqa: E402
from pipelines.common import (build_arg_parser, freeze_modules, run_finetuning, set_random_seeds,  # noqa: E402
                              setup_logging)


def build_model(cfg):
    return LensDetector15(**cfg)


def build_transform(cfg, augment=False):
    return ModelInputTransform(augment=augment, **cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("TEGLIE")
    g.add_argument("--weights", required=True, help="Keras .h5 weights ('Lens Detector 15.h5' or "
                                                    "'Rotated_Flipped_Transposed.h5')")
    g.add_argument("--image_size", type=int, default=101, help="Model input size (weights were trained at 101)")
    g.add_argument("--freeze_stages", type=int, default=0, choices=[0, 1, 2],
                   help="0 = train everything, 1 = freeze CNN backbone, 2 = freeze backbone + transformer encoder")
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--attention_dropout", type=float, default=0.2)
    g.add_argument("--scaling_mode", choices=["robust", "absolute"], default="robust",
                   help="robust: clip each band at its --clip_pct percentile; absolute: original TEGLIE "
                        "clip at +-1e-9 after multiplying by --flux_scale")
    g.add_argument("--clip_pct", type=float, default=99.5)
    g.add_argument("--flux_scale", type=float, default=1.0)
    # Original training: Adam, lr 1e-4 with exponential decay, binary cross-entropy, l2(0.01) on conv/attention
    p.set_defaults(bands="2,3,1,0", learning_rate=1e-4, weight_decay=1e-2, batch_size=64, epochs=50)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "finetune_teglie")

    bands = parse_bands(args.bands)
    if len(bands) != 4:
        raise ValueError(f"TEGLIE expects 4 input bands (r, i, g, u), got --bands {args.bands}")
    model_config = dict(in_channels=4, image_size=args.image_size, dropout=args.dropout,
                        attention_dropout=args.attention_dropout)
    model = load_keras_h5(build_model(model_config), args.weights)
    freeze_modules(model.stage_modules()[:args.freeze_stages])
    logger.info(f"Loaded TEGLIE weights from {args.weights}; frozen stages: {args.freeze_stages}")

    scaling = dict(mode=args.scaling_mode, clip_pct=args.clip_pct) if args.scaling_mode == "robust" \
        else dict(mode="absolute", flux_scale=args.flux_scale)
    transform = build_transform(dict(bands=bands, image_size=args.image_size, resize_mode=args.resize_mode,
                                     scaling="teglie", **scaling), augment=args.augment)
    head_params = [p for m in model.head_modules() for p in m.parameters()]
    run_finetuning(args, "teglie", model, model_config, head_params, transform, logger,
                   extra_info={"pretrained": str(args.weights)})


if __name__ == "__main__":
    main()
