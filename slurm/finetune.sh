#!/bin/bash
#SBATCH --account=ituomine
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

module load python-pytorch/2.13
# srun python3 myprog.py <options>
# Activate your venv ON TOP of the pytorch module (pip/python are wrapped by tykky,

# so packages installed into this venv should still resolve correctly)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1   # use the weights downloaded on the login node

MODEL="deeplens"
SCRATCH_DIR="/scratch/ituomine/COSMOS-WEB-LENS/LensDetection"
MOCK_DATA_DIR="/scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5"
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

srun python3 -m "pipelines.$MODEL" \
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
