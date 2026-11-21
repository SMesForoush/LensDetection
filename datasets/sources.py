"""Collect and split the lens / non-lens files used by train.py and the fine-tuning pipelines.

Data sources (real COSMOS-Web DR1 cutouts, listed in datasets/cosmos_real_manifest.csv, built by
scripts/build_real_manifest.py from the mock-lens-modeling sample lists):
    real_lens_confident    COWLS M25 + S07-S12 lenses                        label 1
    real_lens_calibration  COWLS S03-S06 lenses                              label 1
    lookalike              non-lenses that resemble the COWLS lenses         label 0
    real_nonlens           hand-checked non-lenses                           label 0
PRISM simulations (under --prism_dir):
    prism_lens             <prism_dir>/cosmos_lens_*.npy                     label 1
    prism_nonlens          <prism_dir>/cosmos_nonlens_*.npy                  label 0

Which sources train, validate and test a model is set by --train_data:
    prism  (default) PRISM only. The real manifest is not read at all; real images are kept for the external
           evaluation (`all_real_samples`), which never influences training, checkpoint or threshold choice.
    real   real images only, split by sky region.
    both   real and PRISM images, each split its own way (--include_prism is an old name for this).

Split:
    real   by sky region (0-4, from the project's DR1 tile folds): --test_region -> test, --val_region -> validation,
           the other three regions -> training. A training-region object within 13" of the validation or test region
           (`near_regions`) is dropped, so no training cutout shares pixels with a validation or test cutout.
    PRISM  70/15/15, grouped and stratified (simulations have no sky position). The 25 variations of one simulated
           lens system share a `base_lens_id` in <prism_dir>/cosmos_lens_training_catalog.csv; they always land in the
           same split, so no test lens is a re-render of a training lens. Non-lenses are independent draws, split
           70/15/15 within each `nonlens_mode` (central galaxy / pair / group). The split depends only on --split_seed,
           so runs with different training seeds see the same split.

Real files are found under --cosmos_root (the manifest's `file` column), or by file name in --real_lens_dir /
--lookalike_dir / --real_nonlens_dir when those are given. A PRISM directory may hold its .npy files directly or in
an `npy/` subdirectory.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

COSMOS_ROOT = "/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling"
REAL_MANIFEST = Path(__file__).resolve().parent / "cosmos_real_manifest.csv"
REAL_SOURCES = ("real_lens_confident", "real_lens_calibration", "lookalike", "real_nonlens")
DIR_ARG = {"real_lens_confident": "real_lens_dir", "real_lens_calibration": "real_lens_dir",
           "lookalike": "lookalike_dir", "real_nonlens": "real_nonlens_dir"}

PRISM_DIR = "/scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5"
PRISM_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
TRAIN_DATA_MODES = ("prism", "real", "both")
DEFAULT_TRAIN_DATA = "prism"

# region: sky region 0-4 for real images, -1 for PRISM; near_regions: other regions within 13" (real only)
# group: PRISM split unit (lens: base system, non-lens: the file itself); stratum: PRISM split stratum
Sample = namedtuple("Sample", "path label source region near_regions group stratum", defaults=("", ""))


def add_real_data_arguments(parser):
    """Where the real images are: shared by add_data_arguments and pipelines/evaluate_real.py."""
    parser.add_argument("--cosmos_root", default=COSMOS_ROOT,
                        help="Root that the manifest's file paths are relative to (mock-lens-modeling on Roihu)")
    parser.add_argument("--real_manifest", default=str(REAL_MANIFEST),
                        help="CSV of the real images (scripts/build_real_manifest.py)")
    parser.add_argument("--real_lens_dir", help="Folder of the COWLS lens cutouts (overrides --cosmos_root for lenses)")
    parser.add_argument("--lookalike_dir", help="Folder of the look-alike cutouts (overrides --cosmos_root)")
    parser.add_argument("--real_nonlens_dir", help="Folder of the hand-checked non-lens cutouts (overrides --cosmos_root)")
    return parser


def add_data_arguments(parser):
    """Data arguments shared by train.py and pipelines/."""
    add_real_data_arguments(parser)
    parser.add_argument("--train_data", choices=TRAIN_DATA_MODES,
                        help="Data that trains, validates and tests the model: 'prism' (PRISM simulations only; the "
                             "real images are left for external evaluation), 'real' (real images split by sky region) "
                             f"or 'both'. Default: {DEFAULT_TRAIN_DATA} ('both' with the old --include_prism flag)")
    parser.add_argument("--val_region", type=int, default=4,
                        help="Sky region (0-4) of the real validation set (unused with --train_data prism)")
    parser.add_argument("--test_region", type=int, default=0,
                        help="Sky region (0-4) of the real test set (unused with --train_data prism)")
    parser.add_argument("--include_prism", action="store_true",
                        help="Deprecated: same as --train_data both")
    parser.add_argument("--prism_dir", "--mock_data_dir", dest="prism_dir", default=PRISM_DIR,
                        help="Directory with cosmos_lens_*.npy / cosmos_nonlens_*.npy and the PRISM catalogs")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed of the PRISM train/val/test split; keep it fixed when repeating runs with other --seed values")
    parser.add_argument("--max_samples_per_class", type=int, help="Max samples per data source for balancing")
    return parser


def resolve_train_data(args, logger=None):
    """Fill in args.train_data (None = not given on the command line) and return it.

    The default changed from real data (+ PRISM with --include_prism) to PRISM only. Old command lines keep their
    meaning only if they pass --train_data real, so the default is announced loudly in the log."""
    if args.train_data is None:
        args.train_data = "both" if args.include_prism else DEFAULT_TRAIN_DATA
        if logger:
            if args.include_prism:
                logger.warning("--include_prism is deprecated; using --train_data both")
            else:
                logger.warning(f"--train_data not given: using '{DEFAULT_TRAIN_DATA}' (train, validate and test on "
                               f"PRISM only). Until this change the default was real data; pass --train_data real "
                               f"for the old behaviour.")
    elif args.include_prism and args.train_data == "real":
        raise ValueError("--include_prism contradicts --train_data real; use --train_data both")
    return args.train_data


def uses_real(train_data):
    return train_data in ("real", "both")


def uses_prism(train_data):
    return train_data in ("prism", "both")


def real_dirs(args):
    """{source: folder} from the --real_lens_dir / --lookalike_dir / --real_nonlens_dir overrides."""
    return {s: getattr(args, DIR_ARG[s]) for s in REAL_SOURCES}


def all_real_samples(args, logger=None):
    """Every real image in the manifest, unsplit and unsubsampled (for the external evaluation on real data)."""
    samples = real_samples(args.real_manifest, args.cosmos_root, real_dirs(args), logger)
    if logger:
        counts = pd.Series([s.source for s in samples]).value_counts().to_dict()
        logger.info(f"Real images: {len(samples)} ({sum(s.label for s in samples)} lenses) {counts}")
    return samples


def real_samples(manifest, cosmos_root, dirs=None, logger=None):
    """Real images from the manifest. dirs: {source: folder} overrides cosmos_root for that source."""
    T = pd.read_csv(manifest, dtype={"id": str, "near_regions": str}, keep_default_na=False)
    unknown = set(T.source) - set(REAL_SOURCES)
    if unknown:
        raise ValueError(f"{manifest}: unknown sources {sorted(unknown)}")
    dirs = dirs or {}
    paths = [str(Path(dirs[s]) / Path(f).name) if dirs.get(s) else str(Path(cosmos_root) / f) for s, f in zip(T.source, T.file)]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} real images not found, e.g. {missing[:3]} "
                                f"(set --cosmos_root, or --real_lens_dir / --lookalike_dir / --real_nonlens_dir)")
    return [Sample(p, int(y), s, int(r), str(n)) for p, y, s, r, n in zip(paths, T.label, T.source, T.region, T.near_regions)]


def prism_samples(prism_dir):
    if not prism_dir:
        raise ValueError("PRISM data needs --prism_dir (or use --train_data real)")
    data_dir = Path(prism_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"PRISM directory {data_dir} not found")
    search_dir = data_dir / "npy" if (data_dir / "npy").is_dir() else data_dir
    lens_cat = data_dir / "cosmos_lens_training_catalog.csv"
    if not lens_cat.is_file():
        raise FileNotFoundError(f"{lens_cat} not found: the PRISM split needs each lens's base_lens_id so that variations "
                                f"of one simulated system stay in the same split")
    base_of = dict(pd.read_csv(lens_cat, usecols=["filename_base", "base_lens_id"]).itertuples(index=False))
    nonlens_cat = data_dir / "cosmos_nonlens_training_catalog.csv"
    mode_of = {}
    if nonlens_cat.is_file():
        C = pd.read_csv(nonlens_cat)
        if "nonlens_mode" in C:
            mode_of = dict(zip(C.filename_base, C.nonlens_mode))
    out = []
    for label, pattern, source in ((1, "cosmos_lens_*.npy", "prism_lens"), (0, "cosmos_nonlens_*.npy", "prism_nonlens")):
        files = sorted(search_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"{source}: no files matching '{pattern}' in {search_dir}")
        for f in files:
            if label:
                if f.stem not in base_of:
                    raise ValueError(f"{f.name} is not in {lens_cat.name}, so its base lens system is unknown")
                out.append(Sample(str(f), 1, source, -1, "", f"base{int(base_of[f.stem])}", "lens"))
            else:
                out.append(Sample(str(f), 0, source, -1, "", f.stem, f"nonlens_{mode_of.get(f.stem, 'all')}"))
    return out


def grouped_split(samples, seed, fractions=PRISM_FRACTIONS):
    """Split samples into train/val/test so that every `group` lands in one split, with each `stratum` divided in the
    given fractions (counted in groups). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for stratum in sorted({s.stratum for s in samples}):
        members = [s for s in samples if s.stratum == stratum]
        groups = sorted({s.group for s in members})
        rng.shuffle(groups)
        n_tr = round(fractions[0] * len(groups))
        n_va = round(fractions[1] * len(groups))
        where = {g: 0 if i < n_tr else 1 if i < n_tr + n_va else 2 for i, g in enumerate(groups)}
        for s in members:
            parts[where[s.group]].append(s)
    return parts


def collect_from_args(args, logger=None):
    """Every sample of the sources that --train_data selects, at most --max_samples_per_class per source.
    With --train_data prism the real manifest is not read."""
    mode = resolve_train_data(args, logger)
    samples = []
    if uses_real(mode):
        samples += real_samples(args.real_manifest, args.cosmos_root, real_dirs(args), logger)
    if uses_prism(mode):
        samples += prism_samples(args.prism_dir)
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
        logger.info(f"Total: {len(out)} files, lenses: {n_lens}, non-lenses: {len(out) - n_lens} (--train_data {mode})")
    return out


def split_samples(samples, val_region, test_region, seed, logger=None):
    """Real images by sky region, PRISM images 70/15/15 grouped by base lens system (seed = the PRISM split seed).
    The regions only matter when there are real images; for PRISM-only samples they are ignored.
    Returns {'train'|'val'|'test': (paths, labels, sources)}."""
    has_real = any(s.region >= 0 for s in samples)
    if has_real and (val_region == test_region or not {val_region, test_region} <= set(range(5))):
        raise ValueError(f"--val_region and --test_region must be two different regions in 0-4, got {val_region}, {test_region}")
    parts = {"train": [], "val": [], "test": []}
    dropped = 0
    for s in samples:
        if s.region < 0:
            continue
        if s.region == test_region:
            parts["test"].append(s)
        elif s.region == val_region:
            parts["val"].append(s)
        elif {str(val_region), str(test_region)} & set(s.near_regions.split(";")):
            dropped += 1
        else:
            parts["train"].append(s)
    prism = [s for s in samples if s.region < 0]
    if prism:
        tr, va, te = grouped_split(prism, seed)
        groups = [{s.group for s in part} for part in (tr, va, te)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("A PRISM lens system appears in more than one split!")
        parts["train"] += tr; parts["val"] += va; parts["test"] += te
        if logger:
            n_base = [len({s.group for s in part if s.label}) for part in (tr, va, te)]
            logger.info(f"  PRISM: split seed {seed}, lens base systems train/val/test = {n_base[0]}/{n_base[1]}/{n_base[2]}")
    splits = {k: ([s.path for s in v], [s.label for s in v], [s.source for s in v]) for k, v in parts.items()}
    tr_p, va_p, te_p = (set(splits[k][0]) for k in ("train", "val", "test"))
    if tr_p & va_p or tr_p & te_p or va_p & te_p:
        raise ValueError("Data leak detected between splits!")
    for name, (_, y, _) in splits.items():
        if 0 < sum(y) < len(y):
            continue
        raise ValueError(f"The {name} split needs both classes, found {sum(y)} lenses and {len(y) - sum(y)} non-lenses")
    if logger:
        if has_real:
            logger.info(f"  real images: test = region {test_region}, validation = region {val_region}, training = the "
                        f"other regions; {dropped} training-region images within 13\" of the validation/test regions dropped")
        else:
            logger.info("  no real images: --val_region / --test_region unused")
        for name, (p, y, src) in splits.items():
            counts = pd.Series(src).value_counts().to_dict()
            logger.info(f"  {name:5s}: {len(p)} samples - lens: {sum(y)}, non-lens: {len(y) - sum(y)} {counts}")
    return splits
