#!/usr/bin/env python3
"""Train CMU DeepLens (Lanusse et al. 2018) from scratch, or fine-tune LEXACTUM's weights (Magro et al. 2021).

    python -m pipelines.deeplens --output_dir ... --gpu [--train_data prism|real|both --prism_dir ...]
    python -m pipelines.deeplens --weights weights/lexactum/ground_cmu_deeplens_250epochs.h5 \
        --input_preset native --output_dir ... --gpu

Without --weights (the thesis experiment) the network starts from PyTorch's default initialisation and every layer
is trained; --freeze_stages needs --weights. With --weights the LEXACTUM Keras weights are loaded first.

Weights: https://doi.org/10.5281/zenodo.4299924 (see scripts/download_weights.sh).
    ground_cmu_deeplens_<N>epochs.h5   4 bands (R, I, G, U), Lens Finding Challenge ground-based set
    space_cmu_deeplens_<N>epochs.h5    1 band (Euclid VIS-like), space-based set

Inputs: by default (--input_preset thesis) the shared thesis transform, 4 bands 0,1,2,3 at 300x300 with a per-band
asinh stretch (datasets/transforms.py::THESIS_TRANSFORM). --input_preset native restores LEXACTUM's own input:
bands 2,3,1,0 (R, I, G, U slots) at 101x101, each band ZScale-normalised to [0, 1] (astropy). The dense head depends
on the input size, so at any size other than 101 only the convolutional weights are reused.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import parse_bands  # noqa: E402
from models.deeplens import CMUDeepLens, load_keras_h5  # noqa: E402
from pipelines.common import (build_arg_parser, build_input_transform, freeze_modules,  # noqa: E402
                              parse_args_with_recipe, run_finetuning, set_random_seeds, setup_logging)

NATIVE_BANDS, NATIVE_SIZE = "2,3,1,0", 101  # LEXACTUM: R, I, G, U slots at 101x101


def build_model(cfg):
    return CMUDeepLens(**cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("CMU DeepLens")
    g.add_argument("--weights", help="LEXACTUM Keras .h5 to fine-tune (ground_* for 4 bands, space_* for 1). "
                                     "Default: none, train from random initialisation")
    g.add_argument("--image_size", type=int,
                   help=f"Model input size, --input_preset native only (default {NATIVE_SIZE}, the size of the weights)")
    g.add_argument("--freeze_stages", type=int, default=0, choices=range(7),
                   help="Freeze the stem + first N-1 residual groups (0 = train all, 6 = head only); needs --weights")
    g.add_argument("--dropout", type=float, default=0.0, help="Dropout before the dense head (original: none)")
    g.add_argument("--zscale_contrast", type=float, default=0.25, help="--input_preset native only")
    # From scratch: the shared thesis recipe (C13). With --weights: the finetune recipe plus these values.
    # (Original LEXACTUM training: Adam, lr 1e-3, binary cross-entropy, batch 8.)
    args = parse_args_with_recipe(p, argv, pretrained=lambda a: bool(a.weights),
                                  model_finetune=dict(weight_decay=1e-4, epochs=50))
    if args.freeze_stages and not args.weights:
        p.error("--freeze_stages needs --weights: freezing randomly initialised layers is never wanted")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "finetune_deeplens")

    transform = build_input_transform(args, dict(
        bands=parse_bands(args.bands or NATIVE_BANDS), image_size=args.image_size or NATIVE_SIZE,
        resize_mode=args.resize_mode, scaling="zscale", contrast=args.zscale_contrast), logger)
    model_config = dict(in_channels=len(transform.bands), image_size=transform.image_size, dropout=args.dropout)
    model = build_model(model_config)
    if args.weights:
        skip_head = transform.image_size != NATIVE_SIZE
        if skip_head:
            logger.warning(f"image_size={transform.image_size} != {NATIVE_SIZE}: the dense head is re-initialised")
        if args.input_preset != "native":
            logger.warning("Pretrained weights with the thesis input preset: the weights expect --input_preset native")
        load_keras_h5(model, args.weights, skip_head=skip_head)
        freeze_modules(model.stage_modules()[:args.freeze_stages])
        logger.info(f"Loaded CMU DeepLens weights from {args.weights}; frozen stages: {args.freeze_stages}")
    else:
        logger.info("No --weights: training CMU DeepLens from random initialisation")

    head_params = [p for m in model.head_modules() for p in m.parameters()]
    run_finetuning(args, "deeplens", model, model_config, head_params, transform, logger,
                   extra_info={"pretrained": str(args.weights) if args.weights else None})


if __name__ == "__main__":
    main()
