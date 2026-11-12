#!/usr/bin/env python3
"""Zip the real COSMOS-Web images listed in datasets/cosmos_real_manifest.csv, to train or fine-tune off Roihu.

Run on Roihu (a login node is fine: Python 3 standard library only, nothing is loaded into memory):

    python3 scripts/pack_for_laptop.py --out /scratch/$USER/cosmos_laptop_data.zip
    python3 scripts/pack_for_laptop.py --with_lenses --prism_dir /scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5 \
        --weights_dir weights --out /scratch/$USER/cosmos_laptop_data.zip

By default only the non-lenses are packed (look-alikes and hand-checked non-lenses, about 6.9 GB), because the COWLS
lens cutouts are in mock-lens-modeling/real_cowls_lenses already. Zip layout:
    inputs/dr1_cutouts/...                          non-lenses, at their manifest paths
    data/training/real/real_cowls_lenses/...        lenses (--with_lenses)
    cosmos_real_manifest.csv                        the manifest used
    prism/...                                       PRISM simulations (--prism_dir), as found under that folder
    weights/...                                     pretrained weights (--weights_dir)

On the laptop, after `unzip cosmos_laptop_data.zip -d <data>`:
    python train.py --cosmos_root <data> --real_lens_dir <mock-lens-modeling>/real_cowls_lenses --output_dir ...
(drop --real_lens_dir if the lenses were packed; add --include_prism --prism_dir <data>/prism for PRISM).
The files are float32 noise-dominated images that barely compress, so they are stored uncompressed unless --compress.
"""
import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COSMOS_ROOT = "/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling"
LENS_SOURCES = {"real_lens_confident", "real_lens_calibration"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cosmos_root", default=COSMOS_ROOT, help="Root that the manifest's file paths are relative to")
    ap.add_argument("--manifest", default=str(REPO / "datasets" / "cosmos_real_manifest.csv"))
    ap.add_argument("--out", default="cosmos_laptop_data.zip", help="Zip to write (put it on /scratch, not $HOME)")
    ap.add_argument("--with_lenses", action="store_true", help="Also pack the COWLS lens cutouts of the manifest")
    ap.add_argument("--prism_dir", help="Also pack cosmos_lens_*.npy / cosmos_nonlens_*.npy from this PRISM folder")
    ap.add_argument("--weights_dir", help="Also pack this folder of pretrained weights (scripts/download_weights.sh)")
    ap.add_argument("--compress", action="store_true", help="Deflate the files (slow, saves little on float images)")
    a = ap.parse_args()

    root = Path(a.cosmos_root)
    with open(a.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    # (path on Roihu, path in the zip)
    files = [(root / r["file"], r["file"]) for r in rows if a.with_lenses or r["source"] not in LENS_SOURCES]
    if a.prism_dir:
        prism = Path(a.prism_dir)
        found = sorted(prism.rglob("cosmos_lens_*.npy")) + sorted(prism.rglob("cosmos_nonlens_*.npy"))
        if not found:
            sys.exit(f"no cosmos_lens_*.npy / cosmos_nonlens_*.npy under {prism}")
        files += [(p, "prism/" + p.relative_to(prism).as_posix()) for p in found]
    if a.weights_dir:
        weights = Path(a.weights_dir)
        found = sorted(p for p in weights.rglob("*") if p.is_file() and p.suffix != ".zip")
        if not found:
            sys.exit(f"no files under {weights}")
        files += [(p, "weights/" + p.relative_to(weights).as_posix()) for p in found]

    missing = [str(p) for p, _ in files if not p.is_file()]
    if missing:
        sys.exit(f"{len(missing)} of {len(files)} files not found, e.g.\n  " + "\n  ".join(missing[:5])
                 + "\n(check --cosmos_root and that you can read the folder)")
    total = sum(p.stat().st_size for p, _ in files)
    out = Path(a.out).resolve()
    free = shutil.disk_usage(out.parent).free
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {out} ({free / 1e9:.0f} GB free there)")
    if total > free:
        sys.exit("not enough space for the zip: choose another --out")

    tmp = out.with_name(out.name + ".part")
    method = zipfile.ZIP_DEFLATED if a.compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(tmp, "w", method, allowZip64=True) as z:
        z.write(a.manifest, "cosmos_real_manifest.csv")
        for i, (src, arc) in enumerate(files, 1):
            z.write(src, arc)
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
