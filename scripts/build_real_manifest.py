#!/usr/bin/env python3
"""Build datasets/cosmos_real_manifest.csv: the real COSMOS-Web DR1 images used for training and fine-tuning.

Reads the sample lists of the mock-lens-modeling project (read-only):
    samples/lookalikes/cache/index.csv                    COWLS lenses, look-alikes, hand-checked non-lenses
    samples/benchmark_v1/regions_v2/tile_fold_table.csv   sky region (0-4) and near regions of every object, COWLS tiers
    samples/labels_v2/external_candidates_anomalymatch.csv  AnomalyMatch lens candidates (never a non-lens)
and checks the shape of every lens cutout in --lens_dir.

Rows (source, label):
    real_lens_confident    1  COWLS M25 + S07-S12 (the 79 high-confidence lenses)
    real_lens_calibration  1  COWLS S03-S06 (the 201 medium-confidence lenses)
    lookalike              0  the look-alikes (full / galaxy / arc kinds)
    real_nonlens           0  the hand-checked non-lenses
Left out: COWLS S00-S02 (weak candidates), lens cutouts that are not 300x300 (trimmed at a tile edge), and every
AnomalyMatch candidate ("never a non-lens").

`file` is relative to the project root on Roihu (/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling), the layout
used by the project's own scripts.

    python scripts/build_real_manifest.py --project_dir /path/to/mock-lens-modeling
"""
import argparse
import ast
from pathlib import Path

import pandas as pd

HI = {"M25"} | {f"S{n:02d}" for n in range(7, 13)}
CAL = {"S03", "S04", "S05", "S06"}
LENS_FILE = "data/training/real/real_cowls_lenses/{}_4band_stack.npy"
POOL_FILE = "inputs/dr1_cutouts/pool/{}_4band.npy"
CHECKED_FILE = "inputs/dr1_cutouts/checked_nonlenses/{}_4band.npy"


def npy_shape(path):
    """Array shape from the .npy header (no numpy needed, nothing else is read)."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major = f.read(2)[0]
        n = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        return tuple(ast.literal_eval(f.read(n).decode("latin1"))["shape"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project_dir", required=True, help="mock-lens-modeling folder (needs samples/)")
    ap.add_argument("--lens_dir", help="Folder of the COWLS lens cutouts (default: <project_dir>/real_cowls_lenses, "
                                       "else <project_dir>/data/training/real/real_cowls_lenses)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "datasets" / "cosmos_real_manifest.csv"))
    a = ap.parse_args()
    root = Path(a.project_dir)
    lens_dir = Path(a.lens_dir) if a.lens_dir else next(
        (d for d in (root / "real_cowls_lenses", root / "data/training/real/real_cowls_lenses") if d.is_dir()), None)
    if lens_dir is None:
        raise SystemExit("lens cutout folder not found: give --lens_dir (needed to find trimmed cutouts)")

    index = pd.read_csv(root / "samples/lookalikes/cache/index.csv", dtype={"id": str, "tier": str}, keep_default_na=False)
    folds = pd.read_csv(root / "samples/benchmark_v1/regions_v2/tile_fold_table.csv", dtype={"id": str, "near_folds": str},
                        keep_default_na=False)
    am = pd.read_csv(root / "samples/labels_v2/external_candidates_anomalymatch.csv", dtype={"nearest_54k_id": str},
                     keep_default_na=False)
    am_ids = set(am.nearest_54k_id[am.rule.str.startswith("external candidate")])

    def region_table(kind):
        t = folds[folds.kind == kind].drop_duplicates("id").set_index("id")
        return t.fold.astype(int), t.near_folds

    rows = []
    # ---- lenses: the 79 + S03-S06, full-size cutouts only
    region, near = region_table("cowls_lens")
    tier = folds[folds.kind == "cowls_lens"].drop_duplicates("id").set_index("id").tier
    lenses = index[index.group == "lens"]
    trimmed, weak = [], 0
    for code in lenses.id:
        t = tier[code]
        if t not in HI | CAL:
            weak += 1
            continue
        shape = npy_shape(lens_dir / f"{code}_4band_stack.npy")
        if shape != (4, 300, 300):
            trimmed.append(f"{code} ({t}, {shape})")
            continue
        rows.append(dict(id=code, source="real_lens_confident" if t in HI else "real_lens_calibration", label=1, tier=t,
                         kind="", region=region[code], near_regions=near[code], file=LENS_FILE.format(code)))
    # ---- non-lenses: look-alikes (inference pool) and hand-checked galaxies
    excluded_am = []
    for group, source, kind, template in [("lookalike_full", "lookalike", "full", POOL_FILE),
                                          ("lookalike_galaxy", "lookalike", "galaxy", POOL_FILE),
                                          ("lookalike_arc", "lookalike", "arc", POOL_FILE),
                                          ("training_nonlens", "real_nonlens", "hand-checked", CHECKED_FILE)]:
        region, near = region_table("training_nonlens" if group == "training_nonlens" else "pool")
        for i in index.id[index.group == group]:
            if i in am_ids:
                excluded_am.append(f"{i} ({group})")
                continue
            rows.append(dict(id=i, source=source, label=0, tier="", kind=kind, region=region[i], near_regions=near[i],
                             file=template.format(i)))

    T = pd.DataFrame(rows)
    assert T.file.is_unique, "a file is listed twice"
    assert T.region.between(0, 4).all()
    T.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(T)} rows")
    print(pd.crosstab(T.source, T.region).to_string())
    print(f"left out: {weak} weak COWLS candidates (S00-S02); {len(trimmed)} trimmed lens cutouts: {', '.join(trimmed)}; "
          f"{len(excluded_am)} AnomalyMatch candidates: {', '.join(excluded_am)}")


if __name__ == "__main__":
    main()
