#!/bin/bash
# Zip N random MIRAGE-v1.1 images (half lenses, half non-lenses) as a small example training set.
# Run on Roihu:
#
#   bash scripts/sample_mirage.sh N [out.zip]
#   SEED=1 MIRAGE_DIR=/path/to/mirage_v1_1 bash scripts/sample_mirage.sh 200 /scratch/$USER/mirage_sample.zip
#
# The zip holds lens/<file>.npy (label 1), nonlens/<file>.npy (label 0; companions and hosts),
# index.csv (file,label) and the dataset's lens_catalog.csv / nonlens_catalog.csv when they exist.
# Images are (4, 300, 300) float32 in MJy/sr, about 1.44 MB each. The same SEED picks the same images.
set -euo pipefail

N="${1:?usage: bash scripts/sample_mirage.sh N [out.zip]}"
OUT="${2:-mirage_sample_${N}.zip}"
MIRAGE_DIR="${MIRAGE_DIR:-/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling/datasets/mirage_v1_1}"
SEED="${SEED:-0}"
IMAGES="$MIRAGE_DIR/images"
[ -d "$IMAGES/lens" ] && [ -d "$IMAGES/nonlens" ] || { echo "no lens/ and nonlens/ in $IMAGES"; exit 1; }

# K random .npy names from a folder, the same ones for the same SEED
pick() { ls "$1" | grep '\.npy$' | awk -v s="$SEED" 'BEGIN { srand(s) } { printf "%.12f\t%s\n", rand(), $0 }' | sort | head -n "$2" | cut -f2; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir "$TMP/lens" "$TMP/nonlens"
echo "file,label" > "$TMP/index.csv"
N_LENS=$(( (N + 1) / 2 ))
for kind in lens nonlens; do
    if [ "$kind" = lens ]; then k=$N_LENS; label=1; else k=$(( N - N_LENS )); label=0; fi
    for f in $(pick "$IMAGES/$kind" "$k"); do
        ln -s "$IMAGES/$kind/$f" "$TMP/$kind/$f"
        echo "$kind/$f,$label" >> "$TMP/index.csv"
    done
done
GOT=$(( $(wc -l < "$TMP/index.csv") - 1 ))
[ "$GOT" -eq "$N" ] || { echo "found only $GOT of $N images in $IMAGES"; exit 1; }
for c in lens_catalog.csv nonlens_catalog.csv; do
    [ -f "$IMAGES/$c" ] && cp "$IMAGES/$c" "$TMP/"
done

OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
rm -f "$OUT"
# -0: store only (float images barely compress); zip follows the symlinks and stores the files
(cd "$TMP" && if command -v zip >/dev/null; then zip -q -0 -r "$OUT" .; else python3 -m zipfile -c "$OUT" ./*; fi)
echo "wrote $OUT: $GOT images ($(grep -c ',1$' "$TMP/index.csv") lens), $(du -h "$OUT" | cut -f1)"
