#!/usr/bin/env python3
"""Build datasets/mirage_v1_1_manifest.csv: the MIRAGE-v1.1 images used with --include_mirage.

MIRAGE-v1.1 (mock-lens-modeling/datasets/mirage_v1_1) = real DR1 galaxies (hosts) with a simulated lensed arc added.
Reads (read-only):
    <mirage_dir>/training_rows.csv                         one row per example: kind, item, host_id, region, near_regions
    <mirage_dir>/images/lens/, images/nonlens/             the ready-made (4, 300, 300) MJy/sr images
    samples/benchmark_v1/regions_v2/tile_fold_table.csv    the sky regions of datasets/cosmos_real_manifest.csv
    samples/labels_v2/external_candidates_anomalymatch.csv AnomalyMatch lens candidates (never a non-lens)

Rows (source, label):
    mirage_lens       1  host + lensed arc                       images/lens/<item>.npy
    mirage_companion  0  host + the same source, not lensed      images/nonlens/<item>.npy
    mirage_host       0  the host alone                          images/nonlens/host_<host_id>.npy
Left out: host rows of galaxies already in the real manifest (the same image, it is a look-alike or hand-checked
non-lens there), and host and companion rows on AnomalyMatch candidates. The script stops if an image is missing,
or if a host's sky region differs from the region table of the real manifest, so both splits use the same regions.

    python scripts/build_mirage_manifest.py --project_dir /scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KINDS = {"lens": ("mirage_lens", 1, "images/lens/{item}.npy"),
         "companion": ("mirage_companion", 0, "images/nonlens/{item}.npy"),
         "host": ("mirage_host", 0, "images/nonlens/host_{host_id}.npy")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project_dir", required=True, help="mock-lens-modeling folder")
    ap.add_argument("--mirage_dir", help="MIRAGE dataset folder (default: <project_dir>/datasets/mirage_v1_1)")
    ap.add_argument("--real_manifest", default=str(REPO / "datasets" / "cosmos_real_manifest.csv"))
    ap.add_argument("--out", default=str(REPO / "datasets" / "mirage_v1_1_manifest.csv"))
    a = ap.parse_args()
    root = Path(a.project_dir).resolve()
    mdir = Path(a.mirage_dir).resolve() if a.mirage_dir else root / "datasets" / "mirage_v1_1"

    T = pd.read_csv(mdir / "training_rows.csv", dtype=str, keep_default_na=False)
    print(f"{mdir / 'training_rows.csv'}: {len(T)} rows {T.kind.value_counts().to_dict()}")
    other = T[~T.kind.isin(KINDS)]
    if len(other):
        print(f"left out {len(other)} rows of other kinds (no ready-made image): {other.kind.value_counts().to_dict()}")
    T = T[T.kind.isin(KINDS)].copy()
    bad = ~T.region.isin(list("01234")) | ~T.near_regions.str.fullmatch(r"([0-4](;[0-4])*)?")
    if bad.any():
        sys.exit(f"{bad.sum()} rows with an unexpected region / near_regions, e.g.\n{T[bad].head().to_string()}")
    if (T.groupby("host_id").region.nunique() > 1).any():
        sys.exit("rows of the same host are in different sky regions")

    # ---- the same sky regions as the real manifest
    folds = pd.read_csv(root / "samples/benchmark_v1/regions_v2/tile_fold_table.csv", dtype={"id": str, "near_folds": str},
                        keep_default_na=False)
    ref = folds.drop_duplicates("id").set_index("id")
    hosts = T.drop_duplicates("host_id").set_index("host_id")
    shared = hosts.index.intersection(ref.index)
    split = lambda s: frozenset(x for x in s.split(";") if x)
    diff = [h for h in shared if int(hosts.region[h]) != int(ref.fold[h]) or split(hosts.near_regions[h]) != split(ref.near_folds[h])]
    if diff:
        sys.exit(f"{len(diff)} of {len(shared)} hosts have another region than in tile_fold_table.csv, e.g. "
                 + ", ".join(f"{h}: {hosts.region[h]} [{hosts.near_regions[h]}] vs {ref.fold[h]} [{ref.near_folds[h]}]" for h in diff[:5]))
    print(f"sky regions: {len(shared)} of {len(hosts)} hosts are in tile_fold_table.csv, all with the same region")

    # ---- leave out duplicates of the real non-lenses and anything on an AnomalyMatch candidate
    real_ids = set(pd.read_csv(a.real_manifest, dtype={"id": str}, keep_default_na=False).id)
    am = pd.read_csv(root / "samples/labels_v2/external_candidates_anomalymatch.csv", dtype={"nearest_54k_id": str},
                     keep_default_na=False)
    am_ids = set(am.nearest_54k_id[am.rule.str.startswith("external candidate")])
    dup = (T.kind == "host") & T.host_id.isin(real_ids)
    on_am = T.kind.isin(["host", "companion"]) & T.host_id.isin(am_ids)
    print(f"left out: {dup.sum()} host rows already in the real manifest; {on_am.sum()} host/companion rows on AnomalyMatch candidates")
    T = T[~dup & ~on_am]

    rel = mdir.relative_to(root).as_posix() if mdir.is_relative_to(root) else None
    if rel is None:
        sys.exit(f"--mirage_dir must be inside --project_dir (manifest paths are relative to it): {mdir}")
    rows = []
    for r in T.itertuples():
        source, label, template = KINDS[r.kind]
        rows.append(dict(id=r.item, source=source, label=label, tier="", kind=r.kind, region=int(r.region),
                         near_regions=r.near_regions, file=f"{rel}/{template.format(item=r.item, host_id=r.host_id)}", host_id=r.host_id))
    M = pd.DataFrame(rows)
    assert M.file.is_unique, "a file is listed twice"
    missing = [f for f in M.file if not (root / f).is_file()]
    if missing:
        have = sorted(p.name for p in (mdir / "images/lens").iterdir())[:3] + sorted(p.name for p in (mdir / "images/nonlens").iterdir())[:3]
        sys.exit(f"{len(missing)} of {len(M)} images not found, e.g. {missing[:3]}; files that exist: {have}")
    M.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(M)} rows, {M.host_id.nunique()} hosts")
    print(pd.crosstab(M.source, M.region).to_string())


if __name__ == "__main__":
    main()
