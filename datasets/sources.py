"""Collect and split the lens / non-lens files used by train.py and the fine-tuning pipelines.

Data sources (real COSMOS-Web DR1 cutouts, listed in datasets/cosmos_real_manifest.csv, built by
scripts/build_real_manifest.py from the mock-lens-modeling sample lists):
    real_lens_confident    COWLS M25 + S07-S12 lenses                        label 1
    real_lens_calibration  COWLS S03-S06 lenses                              label 1
    lookalike              non-lenses that resemble the COWLS lenses         label 0
    real_nonlens           hand-checked non-lenses                           label 0
Optional, only with --include_mirage (MIRAGE-v1.1: real DR1 galaxies with a simulated lensed arc added, listed in
datasets/mirage_v1_1_manifest.csv, built by scripts/build_mirage_manifest.py):
    mirage_lens            host galaxy + lensed arc                          label 1
    mirage_companion       host galaxy + the same source, not lensed         label 0
    mirage_host            the host galaxy alone                             label 0

Split by sky region (0-4, from the project's DR1 tile folds), for real and MIRAGE images alike: --test_region -> test,
--val_region -> validation, the other three regions -> training. A training-region image within 13" of the validation
or test region (`near_regions`) is dropped, so no training cutout shares pixels with a validation or test cutout.

Files are found under --cosmos_root (the manifests' `file` column), or by file name in --real_lens_dir /
--lookalike_dir / --real_nonlens_dir when those are given.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

COSMOS_ROOT = "/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling"
REAL_MANIFEST = Path(__file__).resolve().parent / "cosmos_real_manifest.csv"
MIRAGE_MANIFEST = Path(__file__).resolve().parent / "mirage_v1_1_manifest.csv"
REAL_SOURCES = ("real_lens_confident", "real_lens_calibration", "lookalike", "real_nonlens")
MIRAGE_SOURCES = ("mirage_lens", "mirage_companion", "mirage_host")
DIR_ARG = {"real_lens_confident": "real_lens_dir", "real_lens_calibration": "real_lens_dir",
           "lookalike": "lookalike_dir", "real_nonlens": "real_nonlens_dir"}

# region: sky region 0-4; near_regions: other regions within 13", ';'-separated
Sample = namedtuple("Sample", "path label source region near_regions")


def add_data_arguments(parser):
    """Data arguments shared by train.py and pipelines/."""
    parser.add_argument("--cosmos_root", default=COSMOS_ROOT,
                        help="Root that the manifests' file paths are relative to (mock-lens-modeling on Roihu)")
    parser.add_argument("--real_manifest", default=str(REAL_MANIFEST),
                        help="CSV of the real images (scripts/build_real_manifest.py)")
    parser.add_argument("--real_lens_dir", help="Folder of the COWLS lens cutouts (overrides --cosmos_root for lenses)")
    parser.add_argument("--lookalike_dir", help="Folder of the look-alike cutouts (overrides --cosmos_root)")
    parser.add_argument("--real_nonlens_dir", help="Folder of the hand-checked non-lens cutouts (overrides --cosmos_root)")
    parser.add_argument("--val_region", type=int, default=4, help="Sky region (0-4) of the validation set")
    parser.add_argument("--test_region", type=int, default=0, help="Sky region (0-4) of the test set")
    parser.add_argument("--include_mirage", action="store_true",
                        help="Also use the MIRAGE-v1.1 simulations in --mirage_manifest (split by sky region too)")
    parser.add_argument("--mirage_manifest", default=str(MIRAGE_MANIFEST),
                        help="CSV of the MIRAGE images (scripts/build_mirage_manifest.py)")
    parser.add_argument("--max_samples_per_class", type=int, help="Max samples per data source for balancing")
    return parser


def manifest_samples(manifest, sources, cosmos_root, dirs=None):
    """Images listed in a manifest CSV. dirs: {source: folder} overrides cosmos_root for that source."""
    if not Path(manifest).is_file():
        raise FileNotFoundError(f"manifest {manifest} not found")
    T = pd.read_csv(manifest, dtype={"id": str, "near_regions": str}, keep_default_na=False)
    unknown = set(T.source) - set(sources)
    if unknown:
        raise ValueError(f"{manifest}: unknown sources {sorted(unknown)}")
    dirs = dirs or {}
    paths = [str(Path(dirs[s]) / Path(f).name) if dirs.get(s) else str(Path(cosmos_root) / f) for s, f in zip(T.source, T.file)]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} images of {Path(manifest).name} not found, e.g. {missing[:3]} "
                                f"(set --cosmos_root, or --real_lens_dir / --lookalike_dir / --real_nonlens_dir)")
    return [Sample(p, int(y), s, int(r), str(n)) for p, y, s, r, n in zip(paths, T.label, T.source, T.region, T.near_regions)]


def collect_from_args(args, logger=None):
    """Every sample of the requested sources, at most --max_samples_per_class per source."""
    dirs = {s: getattr(args, DIR_ARG[s]) for s in REAL_SOURCES}
    samples = manifest_samples(args.real_manifest, REAL_SOURCES, args.cosmos_root, dirs)
    if args.include_mirage:
        samples += manifest_samples(args.mirage_manifest, MIRAGE_SOURCES, args.cosmos_root)
    by_source = {}
    for s in samples:
        by_source.setdefault(s.source, []).append(s)
    out = []
    for source, group in by_source.items():
        if args.max_samples_per_class and len(group) > args.max_samples_per_class:
            keep = np.random.choice(len(group), args.max_samples_per_class, replace=False)
            group = [group[i] for i in sorted(keep)]
        out += group
        if logger:
            logger.info(f"  {source}: {len(group)} samples")
    n_lens = sum(s.label for s in out)
    if logger:
        logger.info(f"Total: {len(out)} files, lenses: {n_lens}, non-lenses: {len(out) - n_lens}"
                    + ("" if args.include_mirage else " (real data only; --include_mirage adds the MIRAGE simulations)"))
    return out


def split_samples(samples, val_region, test_region, logger=None):
    """Split by sky region. Returns {'train'|'val'|'test': (paths, labels, sources)}."""
    if val_region == test_region or not {val_region, test_region} <= set(range(5)):
        raise ValueError(f"--val_region and --test_region must be two different regions in 0-4, got {val_region}, {test_region}")
    parts = {"train": [], "val": [], "test": []}
    dropped = 0
    for s in samples:
        if s.region == test_region:
            parts["test"].append(s)
        elif s.region == val_region:
            parts["val"].append(s)
        elif {str(val_region), str(test_region)} & set(s.near_regions.split(";")):
            dropped += 1
        else:
            parts["train"].append(s)
    splits = {k: ([s.path for s in v], [s.label for s in v], [s.source for s in v]) for k, v in parts.items()}
    tr_p, va_p, te_p = (set(splits[k][0]) for k in ("train", "val", "test"))
    if tr_p & va_p or tr_p & te_p or va_p & te_p:
        raise ValueError("Data leak detected between splits!")
    for name, (_, y, _) in splits.items():
        if 0 < sum(y) < len(y):
            continue
        raise ValueError(f"The {name} split needs both classes, found {sum(y)} lenses and {len(y) - sum(y)} non-lenses")
    if logger:
        logger.info(f"  test = region {test_region}, validation = region {val_region}, training = the other regions; "
                    f"{dropped} training-region images within 13\" of the validation/test regions dropped")
        for name, (p, y, src) in splits.items():
            counts = pd.Series(src).value_counts().to_dict()
            logger.info(f"  {name:5s}: {len(p)} samples - lens: {sum(y)}, non-lens: {len(y) - sum(y)} {counts}")
    return splits
