#!/bin/bash
#SBATCH --job-name=lens_finetune
#SBATCH --account=ituomine
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/finetune_%x_%j.out
#SBATCH --error=logs/finetune_%x_%j.err

# =============================================================================
# Fine-tune a pretrained lens finder on the base COSMOS-Web dataset (Mahti).
#
#   sbatch slurm/finetune.sh gravit   [extra pipeline args]
#   sbatch slurm/finetune.sh teglie   [extra pipeline args]
#   sbatch slurm/finetune.sh deeplens [extra pipeline args]
#
# Download the weights first, on a login node:  bash scripts/download_weights.sh weights
# =============================================================================
set -euo pipefail

MODEL="${1:?usage: sbatch slurm/finetune.sh <gravit|teglie|deeplens> [args]}"
shift

module purge
module load pytorch   # needs timm >= 1.0 (and transformers for CvT); else: pip install --user -r requirements-finetune.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1   # use the weights downloaded on the login node

SCRATCH_DIR="/scratch/ituomine/cosmos-web-lens/mock-lens-modeling"
MOCK_DATA_DIR="/scratch/ituomine/cosmos-web-lens/custom_20251205_221410_date_20251205_221445"
REAL_LENS_DIR="$SCRATCH_DIR/real_cowls_lenses"
REAL_NONLENS_DIR="$SCRATCH_DIR/filtered_real_nonlens_jwst"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
OUTPUT_DIR="$SCRATCH_DIR/finetune_${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR" logs

case "$MODEL" in
    gravit)
        MODEL_ARGS=(--repo_id parlange/vit-gravit-c3
                    --weights "$WEIGHTS_DIR/gravit/vit-gravit-c3/model.safetensors"
                    --finetune all) ;;
    teglie)
        MODEL_ARGS=(--weights "$WEIGHTS_DIR/teglie/Lens Detector 15.h5") ;;
    deeplens)
        MODEL_ARGS=(--weights "$WEIGHTS_DIR/lexactum/ground_cmu_deeplens_250epochs.h5") ;;
    *)
        echo "Unknown model '$MODEL'"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST: fine-tuning $MODEL -> $OUTPUT_DIR"
nvidia-smi

python3 -m "pipelines.$MODEL" \
    --mock_data_dir "$MOCK_DATA_DIR" \
    --real_lens_dir "$REAL_LENS_DIR" \
    --real_nonlens_dir "$REAL_NONLENS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples_per_class 30000 \
    --lens_weight_multiplier 1.2 \
    --min_specificity 0.8 \
    --gpu \
    --num_workers 6 \
    --seed 42 \
    "${MODEL_ARGS[@]}" "$@"

echo "Done. Best model: $OUTPUT_DIR/best_model.pth"
cat "$OUTPUT_DIR/test_results.json"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State --units=G
