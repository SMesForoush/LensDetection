#!/bin/bash
# Download the pretrained weights used by the fine-tuning pipelines into ./weights (or $1).
# Run on a login node: compute nodes may not have internet access.
#
#   bash scripts/download_weights.sh [weights_dir] [gravit_repo_id ...]
set -euo pipefail

WEIGHTS_DIR="${1:-weights}"
shift || true
GRAVIT_REPOS=("${@:-parlange/vit-gravit-c3}")

mkdir -p "$WEIGHTS_DIR/teglie" "$WEIGHTS_DIR/lexactum" "$WEIGHTS_DIR/gravit"

# --- TEGLIE / Lens Detector 15 (github.com/margres/TEGLIE, TE_weights/) ---
for name in Lens_Detector_15 Rotated_Flipped_Transposed; do
    if [ ! -f "$WEIGHTS_DIR/teglie/$name.zip" ]; then
        curl -fL -o "$WEIGHTS_DIR/teglie/$name.zip" "https://github.com/margres/TEGLIE/raw/main/TE_weights/$name.zip"
    fi
    unzip -oq "$WEIGHTS_DIR/teglie/$name.zip" -d "$WEIGHTS_DIR/teglie"
done

# --- LEXACTUM CMU DeepLens (doi:10.5281/zenodo.4299924) ---
declare -A MD5=(
    [ground_cmu_deeplens_250epochs.h5]=82e0263f5532b80c6d36185756f88838
    [ground_cmu_deeplens_100epochs.h5]=4cd0acb7396bd1e8050263d980a44a30
)
for name in "${!MD5[@]}"; do
    out="$WEIGHTS_DIR/lexactum/$name"
    [ -f "$out" ] || curl -fL -o "$out" "https://zenodo.org/records/4299924/files/$name?download=1"
    echo "${MD5[$name]}  $out" | md5sum -c -
done

# --- GraViT (HuggingFace Hub) ---
for repo in "${GRAVIT_REPOS[@]}"; do
    python3 -c "
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], 'model.safetensors', local_dir=sys.argv[2]))
" "$repo" "$WEIGHTS_DIR/gravit/${repo#*/}"
done

echo "Weights in $WEIGHTS_DIR:"
find "$WEIGHTS_DIR" -name "*.h5" -o -name "*.safetensors"
