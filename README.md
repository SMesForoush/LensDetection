# LensDetection

## Data and training from scratch

`pipelines/lensnet.py` trains the LensNet baseline from scratch, and the other `pipelines/` scripts fine-tune
pretrained models. All of them go through the same runner (`pipelines/common.py::run_finetuning`), read the same data
sources through `datasets/sources.py` and use the same split. `train.py` is the older, separate LensNet loop (see
[LensNet baseline](#lensnet-baseline)); it is kept as legacy and reads the same data.

**Data mode.** `--train_data` sets which images train, validate and test the model:

| `--train_data` | Train / validation / test | Real images |
|---|---|---|
| `prism` (default) | PRISM simulations only, split 70/15/15 by lens system | not read; kept for the external evaluation |
| `real` | real images, split by sky region | all three splits |
| `both` | real + PRISM, each split its own way | all three splits |

The default used to be real data. Old command lines that relied on it now train on PRISM, so pass
`--train_data real` to keep them as they were; the log warns when `--train_data` is not given. `--include_prism` is
an old name for `--train_data both`. `system_info.json` and `test_results.json` record `train_data` and
`split_seed`, and `test_results.json` also has `split_sizes` (images, lenses and sources per split).
`datasets.sources.all_real_samples(args)` returns every usable real image, unsplit, for the evaluation on real data.

**Corrupted files.** Before splitting, `datasets/quality.py` checks every file and drops the ones that cannot be
loaded, cannot be brought to 4×300×300, have more than 50 % NaN/inf pixels, or are all zero. The log prints
`<source>: dropped N corrupted files of M` per source; `data_quality_report.json` lists each dropped file and its
reason, and `test_results.json` has the counts under `dropped_corrupted`. Files with at most 50 % invalid pixels are
kept and inpainted. The check reads each file once and caches the result in `file_qc.json` in the data directory
(or `~/.cache/lensdetection/file_qc/` if that is not writable); a file whose size or modification time changes is
checked again. `LensDatasetMemoryEfficient` raises on a corrupted file instead of returning an all-zero image.

**Real COSMOS-Web DR1 images.** `datasets/cosmos_real_manifest.csv` lists them, with their sky region:

| Source | Images | Label |
|---|---|---|
| `real_lens_confident` | COWLS M25 + S07-S12 lenses: 78 | 1 |
| `real_lens_calibration` | COWLS S03-S06 lenses: 199 | 1 |
| `lookalike` | look-alikes (galaxies that resemble the COWLS lenses): 2,608 | 0 |
| `real_nonlens` | hand-checked non-lenses: 2,146 | 0 |

The files are read from `--cosmos_root` (default `/scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling`, the
manifest's `file` paths are relative to it). `--real_lens_dir`, `--lookalike_dir` and `--real_nonlens_dir` read a
source from another folder instead, by file name.

The manifest is built by `scripts/build_real_manifest.py` from the mock-lens-modeling sample lists
(`samples/lookalikes/cache/index.csv`, `samples/benchmark_v1/regions_v2/tile_fold_table.csv`,
`samples/labels_v2/external_candidates_anomalymatch.csv`). It leaves out the weak COWLS candidates (S00-S02),
the three lens cutouts trimmed at a tile edge (COSJ100024+021749, COSJ095953+022159, COSJ100101+021819) and the
four AnomalyMatch lens candidates that were among the non-lenses (104781, 628424, 87284, 82656).

**Split by sky region** (`--train_data real|both`). DR1 is divided into 5 sky regions. Region `--test_region` (default 0) is the test set,
`--val_region` (default 4) the validation set, and the other three regions are training data. Training-region
images within 13" of the validation or test region are dropped, so no training cutout shares pixels with a
validation or test cutout.

**PRISM simulations** (`--train_data prism|both`). `--prism_dir DIR` (default
`/scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5`) holds `cosmos_lens_*.npy` (`prism_lens`, label 1)
and `cosmos_nonlens_*.npy` (`prism_nonlens`, label 0). They have no sky position and are split 70/15/15, grouped by
`base_lens_id` from `DIR/cosmos_lens_training_catalog.csv`: the 25 variations of one simulated lens system always land
in the same split. Non-lenses are split 70/15/15 within each `nonlens_mode`. The split depends only on `--split_seed`
(default 42), not `--seed`, so repeated runs see the same split: 2,801 / 600 / 599 images (1,400 / 300 / 300
lenses from 56 / 12 / 12 lens systems) at split seed 42. `--val_region` and `--test_region` are unused with
`--train_data prism`. `--mock_data_dir` is accepted as another name for `--prism_dir`.

```bash
python -m pipelines.lensnet --augment --output_dir results/lensnet_prism --gpu              # PRISM only (default)
python -m pipelines.lensnet --augment --train_data real --output_dir results/lensnet_real --gpu
python -m pipelines.lensnet --augment --train_data both --output_dir results/lensnet_real_prism --gpu
sbatch slurm/finetune.sh                       # MODEL (default deeplens) on PRISM only
TRAIN_DATA=real sbatch slurm/finetune.sh       # on real data (TRAIN_DATA=both: real + PRISM)
```

**Running off Roihu.** `scripts/pack_for_laptop.py` zips the manifest's images on Roihu (Python 3 standard library
only). By default it packs the non-lenses, about 6.9 GB uncompressed. `--with_lenses` also packs the lens cutouts,
`--prism_dir` the PRISM files and `--weights_dir` the pretrained weights. The script checks that every file
exists and that the zip will fit before it writes anything.

```bash
# on Roihu, from the repository
python3 scripts/pack_for_laptop.py --prism_dir /scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5 \
    --weights_dir weights --out /scratch/$USER/cosmos_laptop_data.zip
# on the laptop
rsync -P roihu:/scratch/$USER/cosmos_laptop_data.zip . && unzip cosmos_laptop_data.zip -d cosmos_data
python -m pipelines.lensnet --train_data real --cosmos_root cosmos_data \
    --real_lens_dir mock-lens-modeling/real_cowls_lenses --output_dir results/local --gpu
python -m pipelines.lensnet --prism_dir cosmos_data/prism --output_dir results/local_prism --gpu
```

`--gpu` uses an NVIDIA GPU (CUDA) when there is one, and otherwise the Apple Silicon GPU (MPS) on a Mac, in
`train.py` and all the pipelines. Mixed precision stays CUDA-only. On an M-series Mac, CMU DeepLens epochs ran
about 20x faster on the GPU than on the CPU.

## LensNet baseline

`python -m pipelines.lensnet` trains `models/lensnet.py::LensNet` (a five-block VGG-style CNN with global average
pooling, 1,753,570 parameters) from random initialisation. It uses `run_finetuning`, so its loss, optimiser, schedule,
early stopping, validation threshold, outputs and checkpoint are the same as the other pipelines', and
`pipelines/load.py::load_finetuned` reloads it. The input follows decision D4: bands `0,1,2,3` (F115W, F150W, F277W,
F444W), 300 × 300 px, each band scaled to [0, 1] between its 0.5 and 99.5 percentiles and asinh-stretched (softening
0.1). `--scaling none` feeds the raw MJy/sr values as `train.py` did; `--augment` adds random flips and 90°
rotations to the training images. `--dropout` (default 0.5) and the shared training arguments apply.

```bash
python -m pipelines.lensnet --augment --output_dir results/lensnet_prism --gpu
MODEL=lensnet sbatch slurm/finetune.sh
```

`train.py` is the earlier LensNet script, kept as **legacy** so the pilot runs can be reproduced. It has its own loop
(cosine warm restarts, raw MJy/sr input, no augmentation, its own early stopping and plots) and is not comparable with
the pipelines. Its checkpoints still load with `pipelines/load.py`.

**Grad-CAM.** `scripts/gradcam.py` draws Grad-CAM maps of the lens score for images from a predictions CSV
(`test_predictions.csv`, `validation_predictions.csv` or `real_predictions.csv`), using any saved checkpoint and its
own input transform. `--outcomes FP,FN` picks the error types, `--num_samples` how many, and `--layer` the module
(default `pool5`, LensNet's last rectified feature map; other models need `--layer`). It writes
`gradcam_<csv name>.png` and the rows shown as `gradcam_<csv name>.csv`.

```bash
python scripts/gradcam.py results/lensnet_prism --predictions real_predictions.csv --outcomes FP,FN --num_samples 24
```

## Fine-tuning pretrained lens finders

Three pipelines fine-tune published lens-detection models on the base dataset
(`datasets/base.py::LensDatasetMemoryEfficient`, 4 x 300 x 300 cutouts). They use the same file sources,
seed and split as `train.py`, so test results are directly comparable with LensNet.

| Pipeline | Model | Pretrained weights | Model input |
|---|---|---|---|
| `pipelines/gravit.py` | GraViT ViT / DeiT / DeiT3 / CaiT / CvT / Swin / PiT / MLP-Mixer / Twins (Parlange et al. 2025) | HuggingFace `parlange/<family>-gravit-<exp>` | 3 bands, 224 x 224, [0, 1] stretch + ImageNet norm |
| `pipelines/teglie.py` | TEGLIE / Lens Detector 15, CNN + transformer encoder (Thuruthipilly et al. 2022; Grespan et al. 2024) | `Lens Detector 15.h5` (challenge sims) or `Rotated_Flipped_Transposed.h5` (after KiDS fine-tuning) | 4 bands (r, i, g, u), 101 x 101, clipped to [-100, 100] |
| `pipelines/deeplens.py` | LEXACTUM CMU DeepLens (Magro et al. 2021) | Zenodo 4299924, `ground_cmu_deeplens_<N>epochs.h5` | 4 bands (R, I, G, U), 101 x 101, per-band ZScale |

TEGLIE and LEXACTUM are Keras models. They are ported to PyTorch (`models/teglie.py`, `models/deeplens.py`),
and the original `.h5` weights are read with `h5py`, so training does not need TensorFlow.
`scripts/verify_keras_ports.py` rebuilds the original Keras graphs and checks that the ports agree with them
(relative difference about 1e-6 for both TEGLIE weight sets and the DeepLens ground weights).

### Usage

```bash
pip install --user -r requirements-finetune.txt   # if the pytorch module lacks timm/transformers/h5py/astropy
bash scripts/download_weights.sh weights           # on a login node (add GraViT repo ids as extra args)

MODEL=gravit   sbatch slurm/finetune.sh --repo_id parlange/vit-gravit-c3 --weights weights/gravit/vit-gravit-c3/model.safetensors
MODEL=teglie   sbatch slurm/finetune.sh --freeze_stages 1
MODEL=deeplens sbatch slurm/finetune.sh
```

Or run directly, e.g. `python -m pipelines.teglie --help`. Every pipeline accepts the `train.py` data
arguments (`--train_data`, `--cosmos_root`, `--val_region`, `--test_region`, `--prism_dir`,
`--max_samples_per_class`, ...), plus:

- `--bands`: which base-data channels feed the model's input slots. The defaults assume the four base
  channels are ordered blue to red. GraViT uses `3,2,1` (RGB); TEGLIE and DeepLens use `2,3,1,0`, the analogue
  of their r, i, g, u order. Change this if the channel order is different.
- Freezing: GraViT `--finetune head|half|all` (GraViT strategies 1/2/3). TEGLIE `--freeze_stages 0|1|2`
  (none, CNN backbone, backbone + encoder). DeepLens `--freeze_stages 0..6` (stem + residual groups).
- `--learning_rate` (backbone) and `--head_learning_rate`, `--augment` (flips + 90 degree rotations),
  `--resize_mode resize|center_crop`.

### Outputs (in `--output_dir`)

`best_model.pth` (selected on validation F2 with `--min_specificity`), `last_model.pth`, `training_history.json`,
`test_results.json`, `test_predictions.csv`, `validation_predictions.csv`, `system_info.json`,
`data_quality_report.json` (dropped corrupted files), `finetune.log`. LensNet writes the same files.
The decision threshold is tuned for F2 on the validation set and then applied to the test set. It is also stored in
`best_model.pth` (keys `threshold` and `threshold_rule`), by `train.py` too.

Reload a trained model (any pipeline, including LensNet, or a legacy `train.py` checkpoint) together with its exact input transform:

```python
from pipelines.load import load_finetuned, lens_probability
model, transform = load_finetuned("best_model.pth")
p_lens = lens_probability(model, transform, raw_images)   # raw_images: (N, 4, 300, 300)
```

### Evaluation on the real images

`pipelines/evaluate_real.py` scores every real COSMOS-Web image in the manifest (277 lenses + 4,754 non-lenses,
unsplit) with a trained model, at the threshold that the training run chose on its own validation set. The model,
transform and threshold are all read from the checkpoint. There is no threshold option, and no PRISM or validation
data is read. Checkpoints from before the threshold was stored use the threshold in the `test_results.json` next to
them, if that file says it was chosen on validation and its best epoch matches the checkpoint. A model trained with
`--train_data real|both` has seen real images, so the script warns and sets `"external_evaluation": false`.

```bash
python -m pipelines.evaluate_real results/deeplens_prism --gpu     # a run directory, or a .pth file
python -m pipelines.evaluate_real results/deeplens_prism --cosmos_root cosmos_data \
    --real_lens_dir mock-lens-modeling/real_cowls_lenses --gpu      # on the laptop
```

It writes `real_predictions.csv` (same columns as `test_predictions.csv`), `real_results.json`,
`real_data_quality_report.json` (dropped corrupted real files; a separate name, so the training run's
`data_quality_report.json` is kept) and `evaluate_real.log` next to the checkpoint (or to `--output_dir`). `real_results.json` has the metrics at the frozen
threshold plus ROC-AUC and PR-AUC for: `overall`; `per_source` (recall of `real_lens_confident` and
`real_lens_calibration`, specificity and FPR of `lookalike` and `real_nonlens`); and `per_tier` (`confident` = confident
lenses + all non-lenses, `calibration` = calibration lenses + all non-lenses). Next to every PR-AUC are `prevalence`
(lens fraction) and `pr_auc_random`, the PR-AUC of a random classifier, which equals the prevalence.

### Error analysis of the real-data errors

`scripts/error_analysis.py` prepares the structured error analysis (thesis section "Structured error analysis") from
runs that `pipelines.evaluate_real` has scored. Point it at the run directories, or at a folder holding them:

```bash
python scripts/error_analysis.py results --output_dir results/error_analysis --cosmos_root cosmos_data \
    --real_lens_dir mock-lens-modeling/real_cowls_lenses      # on the laptop
```

Runs are grouped by `model_name`. Per architecture it takes the seed with the median real-data ROC-AUC (the lower one
for an even number of seeds), takes all its false negatives and its false positives at the frozen threshold (a random
sample of 150 stratified by source when there are more, `--max_fp`, `--sample_seed`) and writes, to `--output_dir`:
`gallery/<FP|FN>/<id>.png` (asinh RGB with R = F444W, G = F277W, B = (F115W + F150W)/2, plus the four bands, with no
score shown), an empty `coding_sheet.csv` (id, source, error type, `p_lens_<model>` per architecture, primary and
secondary category, notes; hide the score columns while coding), `error_categories.csv`, `sample.csv`,
`selected_runs.csv`, and the cross-model overlap over all errors as `overlap_table.csv` (made by all models vs by only
one) and `error_overlap.csv` (per image). `--skip_gallery` writes the tables without reading any image.

### Notes on the pretrained models

- GraViT: `config.json` names the wrong architecture for pit, mlp-mixer and twins_*. CvT is a HuggingFace
  `transformers` CvT-21, not a timm model. `models/gravit.py` maps each family to the architecture that
  matches the published weights exactly. The ResNet-18 baselines ship only pickles and are not supported.
- TEGLIE was trained on KiDS-like fluxes and clips at +-1e-9. `--scaling_mode robust` (default) clips each band
  at its 99.5th percentile instead; `--scaling_mode absolute --flux_scale <k>` reproduces the original.
- CMU DeepLens' dense head depends on the input size. With `--image_size` other than 101, only the
  convolutional weights are reused.
