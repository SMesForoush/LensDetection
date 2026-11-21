#!/usr/bin/env python3
"""Train the LensNet baseline (models/lensnet.py) from random initialisation with the shared runner.

    python -m pipelines.lensnet --output_dir ... --gpu [--no-augment --train_data prism|real|both --prism_dir ...]

LensNet is a five-block VGG-style CNN (1,753,570 parameters) with global average pooling, so it takes any input size.
It goes through pipelines/common.py::run_finetuning like the other models: same data, split, training recipe (C13:
loss, optimiser, schedule, augmentation, early stopping), validation threshold, outputs and checkpoint format
(pipelines/load.py reloads it).
There are no pretrained weights.

Input (decision D4 in TODO.md): all four bands in their natural order (F115W, F150W, F277W, F444W), 300 x 300 px,
each band of each image scaled to [0, 1] between its 0.5 and 99.5 percentiles and asinh-stretched (softening 0.1).
`--scaling none` feeds the raw MJy/sr values instead, as the legacy train.py did.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform, parse_bands  # noqa: E402
from models.lensnet import LensNet  # noqa: E402
from pipelines.common import (build_arg_parser, parse_args_with_recipe, run_finetuning,  # noqa: E402
                              set_random_seeds, setup_logging)


def build_model(cfg):
    return LensNet(**cfg)


def build_transform(cfg, augment=False):
    return ModelInputTransform(augment=augment, **cfg)


def transform_config(args, bands):
    cfg = dict(bands=bands, image_size=args.image_size, resize_mode=args.resize_mode, scaling=args.scaling)
    if args.scaling == "unit_asinh":
        cfg.update(low_pct=args.low_pct, high_pct=args.high_pct, asinh_softening=args.asinh_softening)
    elif args.scaling == "unit_linear":
        cfg.update(low_pct=args.low_pct, high_pct=args.high_pct)
    return cfg


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("LensNet")
    g.add_argument("--image_size", type=int, default=300, help="Model input size (the cutouts are 300 x 300)")
    g.add_argument("--dropout", type=float, default=0.5, help="Dropout before each dense layer")
    g.add_argument("--scaling", choices=["unit_asinh", "unit_linear", "none"], default="unit_asinh",
                   help="Per-band intensity scaling; 'none' = raw MJy/sr, as in the legacy train.py")
    g.add_argument("--low_pct", type=float, default=0.5, help="Lower percentile mapped to 0 (unit_* scalings)")
    g.add_argument("--high_pct", type=float, default=99.5, help="Upper percentile mapped to 1 (unit_* scalings)")
    g.add_argument("--asinh_softening", type=float, default=0.1, help="asinh softening (unit_asinh)")
    # From scratch, so the training recipe is the shared thesis default (C13); no per-model overrides.
    return parse_args_with_recipe(p, argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "train_lensnet")

    bands = parse_bands(args.bands or "0,1,2,3")
    model_config = dict(num_classes=2, dropout=args.dropout, in_channels=len(bands))
    model = build_model(model_config)
    logger.info("LensNet from random initialisation (no pretrained weights)")

    transform = build_transform(transform_config(args, bands), augment=args.augment)
    head_params = [p for m in (model.fc1, model.fc2, model.fc3) for p in m.parameters()]
    run_finetuning(args, "lensnet", model, model_config, head_params, transform, logger,
                   extra_info={"pretrained": None})


if __name__ == "__main__":
    main()
