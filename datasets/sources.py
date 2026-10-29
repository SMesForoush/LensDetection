"""Collect and split the lens / non-lens files used by train.py and the fine-tuning pipelines.

Data sources:
    mock_lens     <mock_data_dir>/cosmos_lens_*.npy
    mock_nonlens  <mock_data_dir>/cosmos_nonlens_*.npy
    real_lens     <real_lens_dir>/**/*.npy       (optional)
    real_nonlens  <real_nonlens_dir>/**/*.npy    (optional)

`data_mode` selects which sources are used:
    mock       simulated data only; real directories are ignored even if given
    mock+real  simulated + real data; both real directories must exist and contain files
    auto       simulated data, plus each real directory that is given and exists (default)

A directory may hold its .npy files directly or in an `npy/` subdirectory.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_MODES = ("auto", "mock", "mock+real")


def add_data_arguments(parser):
    """Data arguments shared by train.py and pipelines/."""
    parser.add_argument("--mock_data_dir", required=True, help="Directory with cosmos_lens_*.npy / cosmos_nonlens_*.npy")
    parser.add_argument("--real_lens_dir", help="Directory for real lens data (optional)")
    parser.add_argument("--real_nonlens_dir", help="Directory for real non-lens data (optional)")
    parser.add_argument("--data_mode", choices=DATA_MODES, default="auto",
                        help="mock: simulated data only; mock+real: require the real data; "
                             "auto: use the real directories that are given and exist")
    parser.add_argument("--max_samples_per_class", type=int, help="Max samples per data source for balancing")
    return parser


def data_sources(mock_data_dir, real_lens_dir=None, real_nonlens_dir=None, data_mode="auto", logger=None):
    """Return [(directory, label, glob pattern, source name)] for the requested data mode."""
    if data_mode not in DATA_MODES:
        raise ValueError(f"Unknown data_mode '{data_mode}', expected one of {DATA_MODES}")
    sources = [(mock_data_dir, 1, "cosmos_lens_*.npy", "mock_lens"),
               (mock_data_dir, 0, "cosmos_nonlens_*.npy", "mock_nonlens")]
    for directory, label, name in ((real_lens_dir, 1, "real_lens"), (real_nonlens_dir, 0, "real_nonlens")):
        if data_mode == "mock":
            continue
        if data_mode == "mock+real" and not (directory and Path(directory).is_dir()):
            raise FileNotFoundError(f"--data_mode mock+real needs the {name} directory, got: {directory}")
        if directory and not Path(directory).is_dir():
            if logger:
                logger.warning(f"  {name}: directory {directory} not found - skipping (use --data_mode mock "
                               f"to run on simulated data only, or mock+real to make this an error)")
            continue
        if directory:
            sources.append((directory, label, "**/*.npy", name))
    return sources


def collect_files(sources, max_samples_per_class=None, logger=None, required=("mock_lens", "mock_nonlens")):
    """Glob every source. Returns (paths, labels, source names), one entry per file.

    Raises if a source in `required` (or any real source, which is only listed when requested) is empty,
    or if either class ends up with no files.
    """
    paths, labels, origins = [], [], []
    for data_dir, label, pattern, source_name in sources:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"{source_name}: directory {data_dir} not found")
        search_dir = data_dir / "npy" if (data_dir / "npy").is_dir() else data_dir
        files = sorted(search_dir.glob(pattern))
        if not files and (source_name in required or source_name.startswith("real_")):
            raise FileNotFoundError(f"{source_name}: no files matching '{pattern}' in {search_dir}")
        if max_samples_per_class and len(files) > max_samples_per_class:
            files = np.random.choice(files, max_samples_per_class, replace=False).tolist()
        paths.extend(str(f) for f in files)
        labels.extend([label] * len(files))
        origins.extend([source_name] * len(files))
        if logger:
            logger.info(f"  {source_name}: {len(files)} samples from {search_dir}")

    n_lens = sum(labels)
    if n_lens == 0 or n_lens == len(labels):
        raise ValueError(f"Need both classes, found {n_lens} lenses and {len(labels) - n_lens} non-lenses")
    if logger:
        used = sorted(set(origins))
        logger.info(f"Total: {len(paths)} files ({', '.join(used)}), lenses: {n_lens}, "
                    f"non-lenses: {len(labels) - n_lens}")
        if not any(o.startswith("real_") for o in used):
            logger.info("Running on simulated (mock) data only")
    return paths, labels, origins


def collect_from_args(args, logger=None):
    sources = data_sources(args.mock_data_dir, args.real_lens_dir, args.real_nonlens_dir, args.data_mode, logger)
    return collect_files(sources, args.max_samples_per_class, logger)


def split_files(paths, labels, origins, seed, logger=None):
    """Stratified 70/20/10 train/val/test split. Returns {'train'|'val'|'test': (paths, labels, sources)}."""
    tr_p, tmp_p, tr_y, tmp_y, tr_s, tmp_s = train_test_split(
        paths, labels, origins, test_size=0.3, random_state=seed, stratify=labels)
    va_p, te_p, va_y, te_y, va_s, te_s = train_test_split(
        tmp_p, tmp_y, tmp_s, test_size=0.333, random_state=seed, stratify=tmp_y)
    splits = {"train": (tr_p, tr_y, tr_s), "val": (va_p, va_y, va_s), "test": (te_p, te_y, te_s)}
    if set(tr_p) & set(va_p) or set(tr_p) & set(te_p) or set(va_p) & set(te_p):
        raise ValueError("Data leak detected between splits!")
    if logger:
        for name, (p, y, _) in splits.items():
            logger.info(f"  {name:5s}: {len(p)} samples ({100 * len(p) / len(paths):.1f}%) - "
                        f"lens: {sum(y)}, non-lens: {len(y) - sum(y)}")
    return splits
