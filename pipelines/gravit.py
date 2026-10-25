#!/usr/bin/env python3
"""Fine-tune a GraViT model (Parlange et al. 2025) on the base COSMOS-Web dataset.

    python -m pipelines.gravit --repo_id parlange/vit-gravit-c3 --finetune all \
        --mock_data_dir ... --real_lens_dir ... --real_nonlens_dir ... --output_dir ... --gpu

GraViT models were trained on 3-band HSC gri cutouts at 224x224 with ImageNet normalisation, so the 4-band
300x300 base images are reduced to 3 bands (--bands, model RGB order), resized to 224 and stretched to [0, 1].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.transforms import ModelInputTransform, parse_bands  # noqa: E402
from models.gravit import classifier_parameters, load_gravit, set_gravit_trainable  # noqa: E402
from pipelines.common import build_arg_parser, run_finetuning, set_random_seeds, setup_logging  # noqa: E402


def build_transform(cfg, augment=False):
    return ModelInputTransform(augment=augment, **cfg)


def parse_args(argv=None):
    p = build_arg_parser(__doc__)
    g = p.add_argument_group("GraViT")
    g.add_argument("--repo_id", default="parlange/vit-gravit-c3",
                   help="parlange/<family>-gravit-<exp>; families: vit deit deit3 cait cvt swin pit mlp-mixer "
                        "twins_svt twins_pcpvt")
    g.add_argument("--weights", help="Local model.safetensors (default: download from the Hub / HF cache)")
    g.add_argument("--finetune", choices=["head", "half", "all"], default="all",
                   help="GraViT strategy 1 (head only), 2 (half of the layers) or 3 (all layers)")
    g.add_argument("--drop_path_rate", type=float, default=0.1, help="Stochastic depth (GraViT used 0.1)")
    g.add_argument("--stretch", choices=["asinh", "linear"], default="asinh",
                   help="Per-band intensity stretch to [0, 1] before ImageNet normalisation")
    g.add_argument("--low_pct", type=float, default=0.5)
    g.add_argument("--high_pct", type=float, default=99.5)
    p.set_defaults(bands="3,2,1", learning_rate=2e-5, head_learning_rate=1e-3, weight_decay=0.05,
                   batch_size=64, epochs=30, warmup_epochs=2)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, "finetune_gravit")

    bands = parse_bands(args.bands)
    if len(bands) != 3:
        raise ValueError(f"GraViT expects 3 input bands, got --bands {args.bands}")
    model, model_config = load_gravit(args.repo_id, weights=args.weights, drop_path_rate=args.drop_path_rate)
    set_gravit_trainable(model, args.finetune)
    logger.info(f"Loaded {args.repo_id} ({model_config}); fine-tuning mode '{args.finetune}'")

    transform = build_transform(dict(bands=bands, image_size=224, resize_mode=args.resize_mode,
                                     scaling=f"unit_{args.stretch}", imagenet_norm=True,
                                     low_pct=args.low_pct, high_pct=args.high_pct), augment=args.augment)
    run_finetuning(args, "gravit", model, model_config, classifier_parameters(model), transform, logger,
                   extra_info={"pretrained": args.repo_id})


if __name__ == "__main__":
    main()
