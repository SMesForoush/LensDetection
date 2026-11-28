#!/bin/bash
# Submit a models x seeds matrix to Roihu as one Slurm job array (one GPU per (model, seed)), sized automatically.
# Run it from the repository root on a Roihu login node.
#
#   slurm/submit.sh RESULTS [sbatch options] [-- training arguments]
#
#   MODELS="lensnet deeplens efficientnet" SEEDS="0 1 2 3 4" slurm/submit.sh results/roihu/e27
#   MODELS="lensnet deeplens efficientnet" SEEDS=0 slurm/submit.sh results/roihu/smoke --time=0:30:00 \
#       -- --max_samples_per_class 100 --epochs 2                                     # the R26 smoke job
#   TRAIN_DATA=both slurm/submit.sh results/roihu/e29_both --time=8:00:00
#   DRY_RUN=1 slurm/submit.sh results/roihu/e27                                       # print the sbatch command only
#
# Environment: MODELS, SEEDS, SPLIT_SEED, TRAIN_DATA, INPUT_PRESET, PRETRAINED, WEIGHTS_DIR, EVAL_REAL (see
# slurm/finetune.sh), MAX_PARALLEL (at most this many tasks at once; default: no limit).
# Tasks whose run already has test_results.json finish at once, so resubmitting the same command resumes a matrix.
set -euo pipefail

[ $# -ge 1 ] && [ "$1" != "--" ] || { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2; }
RESULTS="$1"; shift
SBATCH_OPTS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do SBATCH_OPTS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift   # the rest are training arguments

[ -f slurm/finetune.sh ] || { echo "submit.sh: run from the repository root" >&2; exit 2; }
MODELS="${MODELS:-lensnet deeplens efficientnet}"
SEEDS="${SEEDS:-0 1 2 3 4}"
read -r -a models <<< "$MODELS"
read -r -a seeds <<< "$SEEDS"
n=$(( ${#models[@]} * ${#seeds[@]} ))
[ "$n" -gt 0 ] || { echo "submit.sh: no models or seeds" >&2; exit 2; }
ARRAY="0-$((n - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}"

echo "submit.sh: ${#models[@]} model(s) [$MODELS] x ${#seeds[@]} seed(s) [$SEEDS] = $n tasks -> $RESULTS"
for i in $(seq 0 $((n - 1))); do
  echo "  task $i: ${models[$(( i / ${#seeds[@]} ))]} seed ${seeds[$(( i % ${#seeds[@]} ))]}"
done
mkdir -p logs "$RESULTS"
# shellcheck disable=SC2054  # the commas belong to --export
cmd=(sbatch --array="$ARRAY" --job-name="lensdet-$(basename "$RESULTS")"
     --export=ALL,RESULTS="$RESULTS",MODELS="$MODELS",SEEDS="$SEEDS"
     ${SBATCH_OPTS[@]+"${SBATCH_OPTS[@]}"} slurm/finetune.sh -- "$@")
printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
"${cmd[@]}"
