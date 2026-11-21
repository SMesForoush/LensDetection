#!/usr/bin/env python3
"""Train EfficientNet-B0 (Tan & Le 2019) from random initialisation.

    python -m pipelines.efficientnet --output_dir ... --gpu [--augment --train_data prism|real|both --prism_dir ...]

The model is timm's `efficientnet_b0` with a 4-channel stem and a 2-logit head (4,010,398 parameters), built with
pretrained=False: there are no ImageNet weights, so the thesis comparison is between architectures trained from
scratch on the same data.

Inputs: the shared thesis transform only (--input_preset thesis, the default): 4 bands 0,1,2,3 at 300x300 with a
per-band asinh stretch (datasets/transforms.py::THESIS_TRANSFORM). EfficientNet has no published lens-finding input of
its own, so --input_preset native is refused.

The training recipe (learning rate, weight decay, epochs, ...) is the shared default of pipelines/common.py; this
script sets no per-model overrides.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.efficientnet import build_efficientnet  # noqa: E402
from pipelines.common import (build_arg_parser, build_input_transform, run_finetuning, set_random_seeds,  # noqa: E402
                              setup_logging)


def build_model(cfg):
    return build_efficientnet(**cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("EfficientNet-B0")
    g.add_argument("--dropout", type=float, default=0.2, help="Dropout before the classifier (EfficientNet-B0: 0.2)")
    g.add_argument("--drop_path_rate", type=float, default=0.2, help="Stochastic depth rate (EfficientNet: 0.2)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.input_preset != "thesis":
        raise SystemExit("EfficientNet-B0 has no native lens-finding input; use --input_preset thesis")
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "train_efficientnet")

    transform = build_input_transform(args, native_config=None, logger=logger)
    model_config = dict(in_channels=len(transform.bands), num_classes=2, dropout=args.dropout,
                        drop_path_rate=args.drop_path_rate)
    model = build_model(model_config)
    logger.info(f"EfficientNet-B0 from random initialisation: {model_config}")

    run_finetuning(args, "efficientnet", model, model_config, list(model.get_classifier().parameters()), transform,
                   logger, extra_info={"pretrained": None})


if __name__ == "__main__":
    main()
