#!/bin/bash
# Copy Roihu results back to the laptop (TODO R26), keeping the run directories intact.
#
#   scripts/sync_roihu.sh                       # all of results/roihu/
#   scripts/sync_roihu.sh results/roihu/e27     # one experiment
#   DRY_RUN=1 scripts/sync_roihu.sh             # list what would be copied
#
# Environment:
#   ROIHU_HOST   ssh host (default: roihu, i.e. a "Host roihu" entry in ~/.ssh/config)
#   ROIHU_REPO   the repository clone on Roihu (default: /scratch/ituomine/COSMOS-WEB-LENS/LensDetection-git)
#   WITH_LAST=1  also copy last_model.pth (by default only best_model.pth, the checkpoint every result comes from)
set -euo pipefail

ROIHU_HOST="${ROIHU_HOST:-roihu}"
ROIHU_REPO="${ROIHU_REPO:-/scratch/ituomine/COSMOS-WEB-LENS/LensDetection-git}"
REL="${1:-results/roihu}"
REL="${REL%/}"
case "$REL" in results/*) ;; *) echo "sync_roihu.sh: give a path under results/, got '$REL'" >&2; exit 2 ;; esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$REL"
opts=(-av --partial)
[ "${WITH_LAST:-0}" = 1 ] || opts+=(--exclude last_model.pth)
[ "${DRY_RUN:-0}" = 1 ] && opts+=(--dry-run)
rsync "${opts[@]}" "$ROIHU_HOST:$ROIHU_REPO/$REL/" "$REL/"
