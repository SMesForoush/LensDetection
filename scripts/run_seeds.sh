#!/usr/bin/env bash
# Multi-seed driver (TODO C16): train every model with every seed, one run after another on the laptop.
#
#   scripts/run_seeds.sh                                    # the E19 runs: lensnet deeplens efficientnet x seeds 0 1 2
#   scripts/run_seeds.sh --real                             # ... and score each run on the real set (C8) afterwards
#   scripts/run_seeds.sh --models lensnet --seeds "0 1" --results results/smoke -- --max_samples_per_class 100 --epochs 2
#   python scripts/aggregate_runs.py results                # mean +- sd per model and set
#
# Each (model, seed) trains with `python -m pipelines.<model>` into <results>/<model>/seed_<s>/. Only --seed changes
# between runs: --split_seed is the same for all of them (default 42), so every seed sees the same train/val/test split.
# Everything after `--` is passed to every training run unchanged (it must not set --seed, --split_seed or
# --output_dir). The data paths default to the laptop (datasets/sources.py), so no path arguments are needed.
#
# Resumable: a (model, seed) whose test_results.json exists is skipped, so the loop can be stopped (Ctrl-C) and started
# again with the same command. A run that was interrupted has no test_results.json yet and is trained again from the
# start. With --real, a run whose real_results.json exists is not scored again. A failed run does not stop the loop;
# the failures are listed at the end and the exit status is 1.
#
# Options:
#   --models "a b ..."   pipelines to train (default: lensnet deeplens efficientnet; also teglie)
#   --seeds "s1 s2 ..."  training seeds (default: 0 1 2, decision D5)
#   --split_seed N       data split seed shared by every run (default: 42)
#   --results DIR        output root (default: results)
#   --real               after training, score each run on every real image (pipelines.evaluate_real)
#   --cpu                do not pass --gpu (default: --gpu, i.e. Apple MPS on the laptop)
#   --dry_run            print the commands without running them
# Environment:
#   PYTHON               interpreter (default: the repo's .venv/bin/python, from a git worktree the main checkout's,
#                        else python3)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="lensnet deeplens efficientnet"
SEEDS="0 1 2"
SPLIT_SEED=42
RESULTS=results
REAL=0
GPU=--gpu
DRY_RUN=0
EXTRA=()

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "run_seeds.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --models) [ $# -ge 2 ] || die "--models needs a value"; MODELS="$2"; shift 2 ;;
    --seeds) [ $# -ge 2 ] || die "--seeds needs a value"; SEEDS="$2"; shift 2 ;;
    --split_seed) [ $# -ge 2 ] || die "--split_seed needs a value"; SPLIT_SEED="$2"; shift 2 ;;
    --results) [ $# -ge 2 ] || die "--results needs a value"; RESULTS="$2"; shift 2 ;;
    --real) REAL=1; shift ;;
    --cpu) GPU=; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown option '$1' (training arguments go after --; see --help)" ;;
  esac
done

for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$arg" in
    --seed|--seed=*|--split_seed|--split_seed=*|--output_dir|--output_dir=*)
      die "$arg is set by the driver; use --seeds, --split_seed or --results instead" ;;
  esac
done
for model in $MODELS; do
  [ -f "$REPO/pipelines/$model.py" ] || die "no pipeline pipelines/$model.py for model '$model'"
done
for seed in $SEEDS $SPLIT_SEED; do
  case "$seed" in ''|*[!0-9]*) die "seeds must be non-negative integers, got '$seed'" ;; esac
done

if [ -z "${PYTHON:-}" ]; then
  # The repo venv; from a git worktree, the venv of the main checkout (worktrees have none of their own).
  MAIN_REPO="$(cd "$REPO" && cd "$(git rev-parse --git-common-dir 2>/dev/null || echo .)/.." 2>/dev/null && pwd)"
  for candidate in "$REPO/.venv/bin/python" "$MAIN_REPO/.venv/bin/python"; do
    if [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
  done
  PYTHON="${PYTHON:-python3}"
fi
case "$RESULTS" in /*) ;; *) RESULTS="$PWD/$RESULTS" ;; esac  # absolute, because the runs start in the repo root

run() {  # print a command, then run it unless --dry_run
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY_RUN" = 1 ] || (cd "$REPO" && "$@")
}

n_models=$(echo $MODELS | wc -w | tr -d ' ')
n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
total=$((n_models * n_seeds))
echo "run_seeds.sh: $n_models model(s) [$MODELS] x $n_seeds seed(s) [$SEEDS] = $total runs, split seed $SPLIT_SEED"
echo "run_seeds.sh: results in $RESULTS, python $PYTHON${EXTRA[0]+, extra arguments: ${EXTRA[*]}}"

i=0
FAILED=()
trap 'echo; echo "run_seeds.sh: interrupted; run the same command again to resume"; exit 130' INT TERM
for model in $MODELS; do
  for seed in $SEEDS; do
    i=$((i + 1))
    dir="$RESULTS/$model/seed_$seed"
    tag="[$i/$total] $model seed $seed"
    if [ -f "$dir/test_results.json" ]; then
      echo "$tag: trained already ($dir/test_results.json), skipping training"
    else
      [ -d "$dir" ] && echo "$tag: $dir has no test_results.json (interrupted run?); training again from the start"
      echo "$tag: training -> $dir ($(date '+%Y-%m-%d %H:%M:%S'))"
      start=$(date +%s)
      [ "$DRY_RUN" = 1 ] || mkdir -p "$dir"
      if ! run "$PYTHON" -m "pipelines.$model" --output_dir "$dir" --seed "$seed" --split_seed "$SPLIT_SEED" \
               $GPU ${EXTRA[@]+"${EXTRA[@]}"}; then
        echo "$tag: training FAILED (log: $dir/finetune.log)"
        FAILED+=("$model seed $seed (training)")
        continue
      fi
      [ "$DRY_RUN" = 1 ] || echo "$tag: trained in $(( ($(date +%s) - start) / 60 )) min"
    fi
    if [ "$REAL" = 1 ]; then
      if [ -f "$dir/real_results.json" ]; then
        echo "$tag: scored on the real set already ($dir/real_results.json), skipping"
      elif ! run "$PYTHON" -m pipelines.evaluate_real "$dir" $GPU; then
        echo "$tag: real-data evaluation FAILED (log: $dir/evaluate_real.log)"
        FAILED+=("$model seed $seed (real-data evaluation)")
      fi
    fi
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "run_seeds.sh: ${#FAILED[@]} step(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "run_seeds.sh: all $total runs done. Aggregate with: $PYTHON scripts/aggregate_runs.py $RESULTS"
