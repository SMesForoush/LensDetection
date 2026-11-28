# shellcheck shell=bash
# Source this on CSC Roihu (login node or inside a job) before running anything from this repository:
#
#   source slurm/roihu_env.sh
#
# It loads the PyTorch module, activates the project venv on top of it (made by slurm/setup_roihu.sh, which adds the
# packages the module lacks, e.g. timm), and points the pipelines at the Roihu copies of the data through the
# LENSDET_* variables that datasets/sources.py reads, so no path arguments are needed (they still override).
#
# Every setting can be overridden from the environment before sourcing, e.g.
#   LENSDET_MODULE=python-pytorch/2.7 source slurm/roihu_env.sh
# Not `set -u` safe on purpose: the module system is not.

LENSDET_PROJECT_DIR="${LENSDET_PROJECT_DIR:-/scratch/ituomine/COSMOS-WEB-LENS}"
LENSDET_MODULE="${LENSDET_MODULE:-python-pytorch}"   # unversioned = CSC's default; the version lands in system_info.json
LENSDET_VENV="${LENSDET_VENV:-$LENSDET_PROJECT_DIR/venv-lensdet}"

module load "$LENSDET_MODULE" || { echo "roihu_env.sh: 'module load $LENSDET_MODULE' failed" >&2; return 1 2>/dev/null || exit 1; }
if [ -f "$LENSDET_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$LENSDET_VENV/bin/activate"
else
  echo "roihu_env.sh: no venv at $LENSDET_VENV yet (run slurm/setup_roihu.sh once on a login node)" >&2
fi

export LENSDET_PRISM_DIR="${LENSDET_PRISM_DIR:-$LENSDET_PROJECT_DIR/prism_jwst_cosmosweb_v5}"
export LENSDET_COSMOS_ROOT="${LENSDET_COSMOS_ROOT:-$LENSDET_PROJECT_DIR/mock-lens-modeling}"  # all four real sources
export PYTHON="${PYTHON:-python3}"          # the interpreter scripts/run_seeds.sh uses
export HF_HUB_OFFLINE=1                     # compute nodes have no internet; weights come from weights/
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-$USER}"
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"; fi
