#!/bin/bash
# One-time setup on a CSC Roihu login node (TODO R26), run from the repository root of a git clone:
#
#   git clone git@github.com:SMesForoush/LensDetection.git /scratch/ituomine/COSMOS-WEB-LENS/LensDetection-git
#   cd /scratch/ituomine/COSMOS-WEB-LENS/LensDetection-git && slurm/setup_roihu.sh
#
# 1. loads the PyTorch module and shows which versions exist,
# 2. makes the project venv on top of it (--system-site-packages) with the packages the module lacks
#    (requirements-finetune.txt: timm, h5py, astropy, ...); torch itself comes from the module,
# 3. checks the imports and the data folders,
# 4. writes data_checksums_roihu.csv and compares it with the laptop's list in git (datasets/data_checksums_laptop.csv).
# Safe to run again: an existing venv is reused and only missing packages are installed.
set -eo pipefail

[ -f slurm/roihu_env.sh ] || { echo "setup_roihu.sh: run from the repository root" >&2; exit 2; }
LENSDET_PROJECT_DIR="${LENSDET_PROJECT_DIR:-/scratch/ituomine/COSMOS-WEB-LENS}"
LENSDET_VENV="${LENSDET_VENV:-$LENSDET_PROJECT_DIR/venv-lensdet}"

echo "== PyTorch modules on this system"
module avail python-pytorch 2>&1 | grep -i pytorch || true

echo "== environment"
# shellcheck disable=SC1091
source slurm/roihu_env.sh || true
if [ ! -f "$LENSDET_VENV/bin/activate" ]; then
  echo "creating venv $LENSDET_VENV on top of $(python3 --version)"
  python3 -m venv --system-site-packages "$LENSDET_VENV"
  # shellcheck disable=SC1091
  source slurm/roihu_env.sh
fi
python3 -m pip install --upgrade-strategy only-if-needed -r requirements-finetune.txt

echo "== imports"
python3 - <<'EOF'
import sys
import astropy, h5py, numpy, pandas, sklearn, timm, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| timm", timm.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
print("GPU visible here:", torch.cuda.is_available(), "(a login node may have none; the jobs do)")
EOF

echo "== data"
for d in "$LENSDET_PRISM_DIR" "$LENSDET_PRISM_DIR/npy" "$LENSDET_COSMOS_ROOT"; do
  if [ -d "$d" ]; then echo "ok   $d"; else echo "MISSING $d"; fi
done
echo "PRISM .npy files: $(find "$LENSDET_PRISM_DIR" -name 'cosmos_*lens_*.npy' 2>/dev/null | wc -l) (laptop: 4,000)"

echo "== checksums (a few minutes)"
python3 scripts/data_checksums.py --output data_checksums_roihu.csv
echo "== compare with the laptop's list (datasets/data_checksums_laptop.csv, in git)"
if python3 scripts/data_checksums.py --compare datasets/data_checksums_laptop.csv data_checksums_roihu.csv; then
  echo "data: identical to the laptop copies"
else
  echo "data: DIFFERS from the laptop copies (see above); fix before any final run" >&2
fi
cat <<EOF

Next:
  1. If the data differ, copy the missing or different files from the laptop (rsync), then run this again.
  2. Smoke job (R26):
       MODELS="lensnet deeplens efficientnet" SEEDS=0 slurm/submit.sh results/roihu/smoke --time=0:30:00 \\
           -- --max_samples_per_class 100 --epochs 2
EOF
