#!/usr/bin/env python3
"""Fine-tune LEXACTUM's CMU DeepLens (Magro et al. 2021; architecture from Lanusse et al. 2018).

    python -m pipelines.deeplens --weights weights/lexactum/ground_cmu_deeplens_250epochs.h5 \
        --mock_data_dir ... --real_lens_dir ... --real_nonlens_dir ... --output_dir ... --gpu

Weights: https://doi.org/10.5281/zenodo.4299924 (see scripts/download_weights.sh).
    ground_cmu_deeplens_<N>epochs.h5   4 bands (R, I, G, U), Lens Finding Challenge ground-based set
    space_cmu_deeplens_<N>epochs.h5    1 band (Euclid VIS-like), space-based set

Inputs are 101x101 with each band ZScale-normalised to [0, 1] (astropy), as in LEXACTUM. The dense head
depends on the input size, so with --image_size other than 101 only the convolutional weights are reused.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform, parse_bands  # noqa: E402
from models.deeplens import CMUDeepLens, load_keras_h5  # noqa: E402
from pipelines.common import (build_arg_parser, freeze_modules, run_finetuning, set_random_seeds,  # noqa: E402
                              setup_logging)


def build_model(cfg):
    return CMUDeepLens(**cfg)


def build_transform(cfg, augment=False):
    return ModelInputTransform(augment=augment, **cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("CMU DeepLens")
    g.add_argument("--weights", required=True, help="LEXACTUM Keras .h5 (ground_* for 4 bands, space_* for 1)")
    g.add_argument("--image_size", type=int, default=101, help="Model input size (weights were trained at 101)")
    g.add_argument("--freeze_stages", type=int, default=0, choices=range(7),
                   help="Freeze the stem + first N-1 residual groups (0 = train all, 6 = head only)")
    g.add_argument("--dropout", type=float, default=0.0, help="Dropout before the dense head (original: none)")
    g.add_argument("--zscale_contrast", type=float, default=0.25)
    # Original training: Adam (lr 1e-3), binary cross-entropy, batch 8
    p.set_defaults(bands="2,3,1,0", learning_rate=1e-4, weight_decay=1e-4, batch_size=64, epochs=50)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "finetune_deeplens")

    bands = parse_bands(args.bands)
    model_config = dict(in_channels=len(bands), image_size=args.image_size, dropout=args.dropout)
    model = build_model(model_config)
    skip_head = args.image_size != 101
    if skip_head:
        logger.warning(f"image_size={args.image_size} != 101: the dense head is re-initialised")
    load_keras_h5(model, args.weights, skip_head=skip_head)
    freeze_modules(model.stage_modules()[:args.freeze_stages])
    logger.info(f"Loaded CMU DeepLens weights from {args.weights}; frozen stages: {args.freeze_stages}")

    transform = build_transform(dict(bands=bands, image_size=args.image_size, resize_mode=args.resize_mode,
                                     scaling="zscale", contrast=args.zscale_contrast), augment=args.augment)
    head_params = [p for m in model.head_modules() for p in m.parameters()]
    run_finetuning(args, "deeplens", model, model_config, head_params, transform, logger,
                   extra_info={"pretrained": str(args.weights)})


if __name__ == "__main__":
    main()
