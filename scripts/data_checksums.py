#!/usr/bin/env python3
"""Checksum list of every data file the experiments read, to check that two machines (laptop, Roihu) hold the same data.

    python scripts/data_checksums.py --output datasets/data_checksums_laptop.csv # on the laptop (kept in git)
    python scripts/data_checksums.py --output data_checksums_roihu.csv           # on Roihu (after slurm/roihu_env.sh)
    python scripts/data_checksums.py --compare datasets/data_checksums_laptop.csv data_checksums_roihu.csv

The list covers the PRISM .npy files and catalogs (under --prism_dir) and every real image in the manifest (resolved
exactly as the training and evaluation code resolves them, so a missing file fails here first). Each row is keyed by
what identifies the file independently of where it lives: the PRISM file name, or the manifest's source and id.
Paths default to the same places as the pipelines (datasets/sources.py, including the LENSDET_* environment variables).

--compare prints files that are missing on either side or differ in content, and exits 1 if there are any.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.sources import PRISM_DIR, add_real_data_arguments, prism_samples, real_dirs, real_samples  # noqa: E402

PRISM_CATALOGS = ("cosmos_lens_training_catalog.csv", "cosmos_nonlens_training_catalog.csv")
FIELDS = ("set", "key", "bytes", "sha256")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def rows(args):
    prism = [Path(s.path) for s in prism_samples(args.prism_dir)]
    prism += [Path(args.prism_dir) / c for c in PRISM_CATALOGS if (Path(args.prism_dir) / c).is_file()]
    real = real_samples(args.real_manifest, args.cosmos_root, real_dirs(args))
    ids = [r["id"] for r in csv.DictReader(open(args.real_manifest))]
    items = [("prism", p.name, p) for p in prism]
    items += [("real", f"{s.source}/{i}", Path(s.path)) for s, i in zip(real, ids)]
    items.append(("manifest", Path(args.real_manifest).name, Path(args.real_manifest)))
    for n, (group, key, path) in enumerate(items, 1):
        if n % 500 == 0:
            print(f"  {n}/{len(items)} files", file=sys.stderr)
        yield {"set": group, "key": key, "bytes": path.stat().st_size, "sha256": sha256(path)}


def compare(a, b):
    def load(p):
        return {(r["set"], r["key"]): r for r in csv.DictReader(open(p))}
    A, B = load(a), load(b)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    differ = sorted(k for k in set(A) & set(B) if A[k]["sha256"] != B[k]["sha256"])
    for label, keys in ((f"only in {a}", only_a), (f"only in {b}", only_b), ("content differs", differ)):
        if keys:
            print(f"{label}: {len(keys)}, e.g. {['/'.join(k) for k in keys[:5]]}")
    same = len(set(A) & set(B)) - len(differ)
    print(f"{same} files identical; {len(only_a)} only in {a}, {len(only_b)} only in {b}, {len(differ)} differ")
    return 1 if (only_a or only_b or differ) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", help="CSV to write the checksum list to")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two checksum lists instead")
    p.add_argument("--prism_dir", default=PRISM_DIR, help="PRISM directory (default as in the pipelines)")
    add_real_data_arguments(p)
    args = p.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.output:
        p.error("give --output (or --compare A B)")
    print(f"PRISM: {args.prism_dir}\nreal: {args.cosmos_root} (lens dir: {real_dirs(args)['real_lens_confident']})",
          file=sys.stderr)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        counts = {}
        for r in rows(args):
            w.writerow(r)
            counts[r["set"]] = counts.get(r["set"], 0) + 1
    print(f"wrote {args.output}: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
