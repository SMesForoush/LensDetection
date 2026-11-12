#!/usr/bin/env python3
"""Zip the COSMOS-Web images listed in the data manifests, to train or fine-tune off Roihu.

Run on Roihu (a login node is fine: Python 3 standard library only, nothing is loaded into memory):

    python3 scripts/pack_for_laptop.py --out /scratch/$USER/cosmos_laptop_data.zip
    python3 scripts/pack_for_laptop.py --with_lenses --with_mirage --mirage_hosts 2000 --weights_dir weights \
        --out /scratch/$USER/cosmos_laptop_data.zip

By default only the real non-lenses of datasets/cosmos_real_manifest.csv are packed (look-alikes and hand-checked
non-lenses, about 6.9 GB), because the COWLS lens cutouts are in mock-lens-modeling/real_cowls_lenses already.
--with_mirage adds the images of datasets/mirage_v1_1_manifest.csv (about 1.44 MB each, roughly 50 GB in all);
--mirage_hosts N keeps the lenses, companions and hosts of N random host galaxies only. Zip layout:
    inputs/dr1_cutouts/...                          real non-lenses, at their manifest paths
    data/training/real/real_cowls_lenses/...        real lenses (--with_lenses)
    datasets/mirage_v1_1/images/...                 MIRAGE images (--with_mirage)
    cosmos_real_manifest.csv, mirage_v1_1_manifest.csv   the manifests of what was packed
    weights/...                                     pretrained weights (--weights_dir)

On the laptop, after `unzip cosmos_laptop_data.zip -d <data>`:
    python train.py --cosmos_root <data> --real_lens_dir <mock-lens-modeling>/real_cowls_lenses --output_dir ...
(drop --real_lens_dir if the lenses were packed; add --include_mirage --mirage_manifest <data>/mirage_v1_1_manifest.csv
for MIRAGE). The files are float32 noise-dominated images that barely compress, so they are stored uncompressed
unless --compress.
"""
import argparse
import csv
import io
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COSMOS_ROOT = "/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling"
LENS_SOURCES = {"real_lens_confident", "real_lens_calibration"}


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def to_csv(fields, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cosmos_root", default=COSMOS_ROOT, help="Root that the manifests' file paths are relative to")
    ap.add_argument("--manifest", default=str(REPO / "datasets" / "cosmos_real_manifest.csv"))
    ap.add_argument("--mirage_manifest", default=str(REPO / "datasets" / "mirage_v1_1_manifest.csv"))
    ap.add_argument("--out", default="cosmos_laptop_data.zip", help="Zip to write (put it on /scratch, not $HOME)")
    ap.add_argument("--with_lenses", action="store_true", help="Also pack the COWLS lens cutouts of the manifest")
    ap.add_argument("--with_mirage", action="store_true", help="Also pack the MIRAGE-v1.1 images (--mirage_manifest)")
    ap.add_argument("--mirage_hosts", type=int, help="Pack the MIRAGE images of this many random host galaxies only")
    ap.add_argument("--seed", type=int, default=0, help="Seed of the --mirage_hosts draw")
    ap.add_argument("--weights_dir", help="Also pack this folder of pretrained weights (scripts/download_weights.sh)")
    ap.add_argument("--compress", action="store_true", help="Deflate the files (slow, saves little on float images)")
    a = ap.parse_args()

    root = Path(a.cosmos_root)
    _, rows = read_csv(a.manifest)
    # (path on Roihu, path in the zip)
    files = [(root / r["file"], r["file"]) for r in rows if a.with_lenses or r["source"] not in LENS_SOURCES]
    extra = {"cosmos_real_manifest.csv": open(a.manifest).read()}
    if a.with_mirage:
        if not Path(a.mirage_manifest).is_file():
            sys.exit(f"{a.mirage_manifest} not found: build it with scripts/build_mirage_manifest.py")
        fields, mrows = read_csv(a.mirage_manifest)
        if a.mirage_hosts:
            hosts = sorted({r["host_id"] for r in mrows})
            keep = set(random.Random(a.seed).sample(hosts, min(a.mirage_hosts, len(hosts))))
            mrows = [r for r in mrows if r["host_id"] in keep]
        files += [(root / r["file"], r["file"]) for r in mrows]
        extra["mirage_v1_1_manifest.csv"] = to_csv(fields, mrows)
        print(f"MIRAGE: {len(mrows)} images of {len({r['host_id'] for r in mrows})} hosts")
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
        for name, text in extra.items():
            z.writestr(name, text)
        for i, (src, arc) in enumerate(files, 1):
            z.write(src, arc)
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
