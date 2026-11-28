#!/bin/bash
#SBATCH --job-name=lensdet
#SBATCH --account=ituomine
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/lensdet_%A_%a.out
#SBATCH --error=logs/lensdet_%A_%a.err
#
# One training run on CSC Roihu (1x GH200), or one task of a job array: task i trains model i / n_seeds with seed
# i % n_seeds, through scripts/run_seeds.sh, so the run directory layout (<RESULTS>/<model>/seed_<s>/), the shared
# thesis recipe (C13), the resume rule (a run with test_results.json is skipped) and the real-data evaluation (C8) are
# exactly those of the laptop runs. Submit arrays with slurm/submit.sh, which sizes --array for you:
#
#   MODELS="lensnet deeplens efficientnet" SEEDS="0 1 2 3 4" slurm/submit.sh results/roihu/e27
#   MODEL=deeplens SEED=0 RESULTS=results/roihu/test sbatch slurm/finetune.sh -- --epochs 2   # a single run
#
# Settings (environment):
#   RESULTS        output root, relative to the repository (required)
#   MODELS/MODEL   pipelines (default: lensnet deeplens efficientnet)
#   SEEDS/SEED     training seeds (default: 0 1 2 3 4)
#   SPLIT_SEED     data split seed shared by every run (default 42; keep it fixed)
#   TRAIN_DATA     prism (default) | real | both, see datasets/sources.py
#   INPUT_PRESET   thesis (default) | native (published inputs; pretrained side experiment; forced for gravit)
#   PRETRAINED     0 (default: random init) | 1 (published Keras weights for teglie/deeplens; side experiment)
#   WEIGHTS_DIR    where the published weights are (default: weights, see scripts/download_weights.sh)
#   EVAL_REAL      1 (default): score every real image after training (pipelines.evaluate_real); 0 to skip
# Arguments after `--` go to every training run unchanged (not --seed, --split_seed or --output_dir).
# Data paths and the Python environment come from slurm/roihu_env.sh.
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR" || exit 2; fi
REPO="$(pwd)"
[ -f "$REPO/scripts/run_seeds.sh" ] || { echo "finetune.sh: submit from the repository root" >&2; exit 2; }
[ "${1:-}" = "--" ] && shift

# shellcheck disable=SC1091
source "$REPO/slurm/roihu_env.sh" || exit 2

: "${RESULTS:?set RESULTS, e.g. RESULTS=results/roihu/e27}"
read -r -a models <<< "${MODELS:-${MODEL:-lensnet deeplens efficientnet}}"
read -r -a seeds <<< "${SEEDS:-${SEED:-0 1 2 3 4}}"
n=$(( ${#models[@]} * ${#seeds[@]} ))
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$i" -ge "$n" ]; then echo "finetune.sh: task $i but only $n (model, seed) pairs" >&2; exit 2; fi
model="${models[$(( i / ${#seeds[@]} ))]}"
seed="${seeds[$(( i % ${#seeds[@]} ))]}"

TRAIN_DATA="${TRAIN_DATA:-prism}"
INPUT_PRESET="${INPUT_PRESET:-thesis}"
PRETRAINED="${PRETRAINED:-0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
if [ "$model" = gravit ]; then INPUT_PRESET=native; fi

MODEL_ARGS=()
case "$model" in
  lensnet|efficientnet) ;;   # always from scratch
  gravit)
    MODEL_ARGS=(--repo_id parlange/vit-gravit-c3 --weights "$WEIGHTS_DIR/gravit/vit-gravit-c3/model.safetensors"
                --finetune all) ;;
  teglie)
    if [ "$PRETRAINED" = 1 ]; then MODEL_ARGS=(--weights "$WEIGHTS_DIR/teglie/Lens Detector 15.h5"); fi ;;
  deeplens)
    if [ "$PRETRAINED" = 1 ]; then MODEL_ARGS=(--weights "$WEIGHTS_DIR/lexactum/ground_cmu_deeplens_250epochs.h5"); fi ;;
  *) echo "finetune.sh: unknown model '$model'" >&2; exit 2 ;;
esac

REAL_FLAG=(--real)
if [ "${EVAL_REAL:-1}" = 0 ]; then REAL_FLAG=(); fi

echo "Job ${SLURM_JOB_ID:-local} task $i/$n on ${SLURM_NODELIST:-$(hostname)}: $model seed $seed" \
     "(train_data=$TRAIN_DATA, input_preset=$INPUT_PRESET, pretrained=$PRETRAINED) -> $RESULTS/$model/seed_$seed"
echo "git $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (with local changes)')"
module list 2>&1 | tail -n +1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv

"$REPO/scripts/run_seeds.sh" --models "$model" --seeds "$seed" --split_seed "${SPLIT_SEED:-42}" \
  --results "$RESULTS" "${REAL_FLAG[@]}" -- \
  --train_data "$TRAIN_DATA" --input_preset "$INPUT_PRESET" "${MODEL_ARGS[@]}" "$@"
status=$?

if [ -n "${SLURM_JOB_ID:-}" ]; then
  sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State --units=G 2>/dev/null
fi
exit $status
