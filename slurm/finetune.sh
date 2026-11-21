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

MODEL="${MODEL:-deeplens}"   # lensnet | deeplens | teglie | gravit | efficientnet, e.g. MODEL=lensnet sbatch slurm/finetune.sh
SCRATCH_DIR="/scratch/ituomine/COSMOS-WEB-LENS/LensDetection"
# TRAIN_DATA picks the data that trains, validates and tests the model (datasets/sources.py):
#   prism (default)  PRISM simulations only; the real images are kept for the external evaluation
#   real             real images (COWLS lenses, look-alikes, hand-checked non-lenses), split by sky region
#   both             real + PRISM
# e.g.  TRAIN_DATA=real sbatch slurm/finetune.sh   (INCLUDE_PRISM=1 is the old spelling of TRAIN_DATA=both)
# Real images are read from the mock-lens-modeling project, paths in datasets/cosmos_real_manifest.csv.
COSMOS_ROOT="/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling"
PRISM_DIR="/scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5"
if [ "${INCLUDE_PRISM:-0}" = 1 ] && [ -z "${TRAIN_DATA:-}" ]; then TRAIN_DATA=both; fi
TRAIN_DATA="${TRAIN_DATA:-prism}"
# INPUT_PRESET picks the input transform (datasets/transforms.py):
#   thesis (default)  the one transform shared by every architecture: bands 0,1,2,3, 300x300, per-band unit_asinh
#   native            the model's own published input (TEGLIE/DeepLens 2,3,1,0 at 101 px), for the pretrained side
#                     experiment. GraViT only supports native; EfficientNet only supports thesis.
# e.g.  INPUT_PRESET=native sbatch slurm/finetune.sh
INPUT_PRESET="${INPUT_PRESET:-thesis}"
if [ "$MODEL" = gravit ]; then INPUT_PRESET=native; fi
# PRETRAINED picks the starting weights of TEGLIE and DeepLens (GraViT always loads its weights; LensNet and
# EfficientNet never do):
#   0 (default)  random initialisation, all layers trained (the thesis experiment)
#   1            load the published Keras weights from WEIGHTS_DIR (pretrained side experiment; pair it with
#                INPUT_PRESET=native, the input those weights expect)
# e.g.  PRETRAINED=1 INPUT_PRESET=native sbatch slurm/finetune.sh
PRETRAINED="${PRETRAINED:-0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
OUTPUT_DIR="$SCRATCH_DIR/finetune_${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR" logs

case "$MODEL" in
    lensnet)
        MODEL_ARGS=(--augment) ;;   # from scratch, no weights
    gravit)
        MODEL_ARGS=(--repo_id parlange/vit-gravit-c3
                    --weights "$WEIGHTS_DIR/gravit/vit-gravit-c3/model.safetensors"
                    --finetune all) ;;
    teglie)
        MODEL_ARGS=()
        if [ "$PRETRAINED" = 1 ]; then MODEL_ARGS=(--weights "$WEIGHTS_DIR/teglie/Lens Detector 15.h5"); fi ;;
    deeplens)
        MODEL_ARGS=()
        if [ "$PRETRAINED" = 1 ]; then MODEL_ARGS=(--weights "$WEIGHTS_DIR/lexactum/ground_cmu_deeplens_250epochs.h5"); fi ;;
    efficientnet)  # EfficientNet-B0 from random initialisation, no weights
        MODEL_ARGS=() ;;
    *)
        echo "Unknown model '$MODEL'"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST: training $MODEL (train_data=$TRAIN_DATA, input_preset=$INPUT_PRESET, pretrained=$PRETRAINED) -> $OUTPUT_DIR"
nvidia-smi

srun python3 -m "pipelines.$MODEL" \
    --train_data "$TRAIN_DATA" \
    --input_preset "$INPUT_PRESET" \
    --cosmos_root "$COSMOS_ROOT" \
    --prism_dir "$PRISM_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples_per_class 30000 \
    --lens_weight_multiplier 1.2 \
    --gpu \
    --num_workers 6 \
    --seed 42 \
    "${MODEL_ARGS[@]}" "$@"

echo "Done. Best model: $OUTPUT_DIR/best_model.pth"
cat "$OUTPUT_DIR/test_results.json"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State --units=G
