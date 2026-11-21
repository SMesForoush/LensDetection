# Thesis Chapter 5: open work

This is the work still needed before the training and evaluation code matches the thesis design (Chapter 5, "Methods:
Training and Evaluation of Lens Classifiers"). Each item can be picked up on its own: it says what exists now, what to
change, which files to touch, what it depends on and how to check it is done.

**Working on an item:** set its `Status` to `in progress (<branch>)` when you start, and to `done (<PR link>)` when the
PR is open. If you find the item's description no longer matches the code, fix the description here in the same PR.

Last updated 2026-11-21 (Roihu dropped: every run is now on the laptop, see "Where runs happen").

---

## Context every agent needs

### The thesis design (what the code must end up doing)

1. **Train only on PRISM simulations, from random initialisation.** Use no pretrained weights in the main experiment.
2. **Fix all model selection on synthetic data.** Early stopping, checkpoint choice and the decision threshold use only
   the PRISM validation split.
3. **Use the real COSMOS-Web set only for external evaluation**, once, with the frozen model and frozen threshold.
   No real image may influence training, checkpoint selection or the threshold.
4. **Compare three architectures:** a simple CNN baseline (LensNet), CMU DeepLens (ResNet) and EfficientNet-B0.
   TEGLIE is optional. GraViT is excluded.
5. **Train each architecture with 3–5 seeds** and report mean ± sd, plus bootstrap 95 % intervals over test images.
6. **Do a structured error analysis** of false positives and false negatives on the real set.

**What the code does today:** it trains, validates and tests on PRISM only by default (`--train_data prism`, C6), and
feeds every model the same input (`--input_preset thesis`, C9). DeepLens and TEGLIE train from random initialisation
unless `--weights` is given (C11), and every from-scratch run uses one training recipe (C13). `--train_data real|both`
keeps the old real-data split by sky region. Most items below exist to close the remaining gap.

### Where runs happen

**Everything runs on the laptop** (Apple Silicon, `--gpu` = MPS): smoke tests, the final multi-seed runs and the
real-data evaluation. There is no budget for CSC Roihu, so do not plan, size or prepare anything for it. New items do
not need to touch `slurm/finetune.sh` or `scripts/pack_for_laptop.py`; they are left as they are, unmaintained. O26 is
the optional task for reviving Roihu later.

The data paths default to the laptop (`datasets/sources.py`: `PRISM_DIR`, `COSMOS_ROOT`, `REAL_LENS_DIR`), anchored at
`~/aalto/thesis`, so runs need no path arguments, also from a git worktree. The Roihu paths are kept as
`ROIHU_PRISM_DIR` and `ROIHU_COSMOS_ROOT`.

### Data

| Set | Content | Where |
|---|---|---|
| PRISM (synthetic) | 2,000 lenses = 80 base systems (`base_lens_id`) × 25 variations; 2,000 non-lenses (688 `central_galaxy`, 698 `galaxy_pair`, 614 `galaxy_group`). Shape (4, 300, 300) float32, MJy/sr, 0.03″/px, bands F115W, F150W, F277W, F444W = channels 0–3. Einstein radius 0.30–2.11″, median 0.53″. | Laptop: `/Users/sahel-smartly/aalto/thesis/prism/out/prism` (`npy/` + `cosmos_lens_training_catalog.csv` + `cosmos_nonlens_training_catalog.csv`); the `--prism_dir` default. (Roihu, not used: `/scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5`) |
| Real (COSMOS-Web DR1) | 277 lenses (`real_lens_confident` 78: COWLS M25 + S07–S12; `real_lens_calibration` 199: S03–S06) and 4,754 non-lenses (`lookalike` 2,608 hard negatives; `real_nonlens` 2,146 hand-checked). Same format as PRISM. | `datasets/cosmos_real_manifest.csv` (paths relative to `--cosmos_root`). Laptop (the defaults): `--cosmos_root cosmos_data` (look-alikes, non-lenses), `--real_lens_dir mock-lens-modeling/real_cowls_lenses` (lenses). (Roihu, not used: `--cosmos_root /scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling`) |

The current real-data split by sky region (`--val_region 4 --test_region 0`) gives train 3,073 images (168 lenses),
val 973 (56), test 940 (48), with 45 dropped.

### Code map

| File | Role |
|---|---|

| `datasets/sources.py` | Collects samples (`collect_from_args`) and splits them (`split_samples`). `--train_data prism` (default) \| `real` \| `both` picks the sources. PRISM is split 70/15/15, grouped by `base_lens_id`, under `--split_seed` (done in #7); real data by region. `all_real_samples(args)` returns every real image, unsplit, for C8. Both return `(samples, quality_report)`: corrupted files are dropped first (C14). |
| `datasets/quality.py` | Finds corrupted files (unreadable, wrong shape, > 50 % NaN/inf, all zero) before training, cached per data directory in `file_qc.json` (C14). |
| `pipelines/evaluate_real.py` | C8: scores all real images with a checkpoint at its frozen validation threshold and writes `real_predictions.csv` + `real_results.json` (with bootstrap `ci95`, C15) + `real_{roc,pr}_curve.csv`. |
| `scripts/error_analysis.py` | C17: picks the median-ROC-AUC seed per architecture, samples its real-data FP/FN, renders a blind gallery and writes an empty coding sheet plus the cross-model overlap table. |
| `datasets/base.py` | `LensDatasetMemoryEfficient`: loads `.npy`, pads or crops to 4×300×300, inpaints NaN/inf, clips at ±10× the 99.9th percentile of \|x\|. |

| `datasets/transforms.py` | `ModelInputTransform`: band selection → resize → intensity scaling (`unit_asinh`, `unit_linear`, `teglie`, `zscale`, `none`) → optional ImageNet norm → optional D4 augmentation. `ModelInputTransform.thesis_default()` is the shared D4 input (`--input_preset thesis`, default; C9). |
| `pipelines/common.py` | Shared runner `run_finetuning`, plus `binary_metrics` (with FPR and prevalence), `bootstrap_ci` and `save_curves` (C15), `best_f2_threshold`, `save_predictions` and `build_arg_parser`, whose defaults are the thesis training recipe (C13); `parse_args_with_recipe` switches pretrained runs to `FINETUNE_RECIPE`. |
| `pipelines/lensnet.py`, `deeplens.py`, `teglie.py`, `gravit.py`, `efficientnet.py` | One script per model, built on `run_finetuning`. LensNet (C10) and EfficientNet-B0 (C12) train from scratch with the D4 input. DeepLens and TEGLIE train from scratch without `--weights` (C11); GraViT always loads its weights. |
| `pipelines/load.py` | Rebuilds a saved model (pipelines or `train.py` LensNet) and its transform from `best_model.pth`; `frozen_threshold` reads its validation threshold. |
| `train.py` | **Legacy** LensNet loop that does not use `run_finetuning` (kept to reproduce the pilot runs). Use `pipelines/lensnet.py`. |
| `scripts/gradcam.py` | Grad-CAM of the lens score for rows of a predictions CSV, from any checkpoint (moved out of `train.py` in C10, for E21). |
| `models/*.py` | Architectures: `lensnet.py` (1,753,570 params), `deeplens.py` (3,557,281 at 300 px), `teglie.py` (3,193,889), `efficientnet.py` (timm EfficientNet-B0, 4,010,398), `gravit.py`. |
| `slurm/finetune.sh` | Roihu job (1× GH200). **Not used**: kept unmaintained, see O26. New items do not need to update it. |
| `../Master-thesis/sections/*.tex` | Thesis text (`data.tex`, `methodology.tex`, `literature.tex`, `introduction.tex`). |

### Conventions

- Use the repo venv: `.venv/bin/python` (Apple MPS with `--gpu`, Python 3.14, PyTorch 2.15). All runs, final ones
  included, happen here.
- For smoke tests, use the local PRISM copy with `--max_samples_per_class 100 --epochs 2`.
- **Never use test labels for any choice** (threshold, checkpoint, hyperparameter), and never use real-data labels for
  anything except the final evaluation.
- Keep `--split_seed` fixed (default 42) across all final runs. Only `--seed` varies between repetitions.
- Every run writes `system_info.json`, `training_history.json`, `test_results.json`, `data_quality_report.json` and
  `{validation,test}_predictions.csv` to `--output_dir`. Keep that contract, because later items read those files.

### Dependency order

```
Decisions D1–D5 (supervisor) ──► C6 PRISM-only mode ──► C8 real-data evaluation ──► E20 ──► E21
                                  C9 shared transform ─┐
                                  C10 LensNet → shared runner + augmentation
                                  C11 DeepLens from scratch, C12 EfficientNet ─┤
                                  C13 shared recipe ───┘──► E18 smoke test ──► E19 final runs on the laptop (needs C16)
C14, C15, C16, C17 can be done in parallel at any time. O26 (Roihu) is optional and blocks nothing.
```

---

## Decisions (confirm with supervisor before coding the parts they affect)

### D1. Train on PRISM only; use real data only for evaluation
- **Status:** Done
- **Affects:** C6, C8, C13, E19, W22, W24
- **Question:** Confirm that the final experiment trains and validates only on PRISM and evaluates once on all real
  cutouts. The thesis RQs (`introduction.tex`, RQ1–RQ4) already assume this; the code currently does the opposite.
- **Default if no answer:** yes. Everything below assumes it.
- **Decision:** Will do the final experiments only on PRISM. keep the rest of experiments as just side experiments. 

### D2. What the baseline is
- **Status:** Done
- **Affects:** C10, W24
- **Question:** `literature.tex` says the baseline is "modelled on the compact, published CNN of Schaefer et al. (2018)".
  LensNet (`models/lensnet.py`) is a generic five-block VGG-style CNN, not that network. Either keep LensNet and
  rewrite the sentence, or port the Schaefer et al. CNN.
- **Decision:** keep LensNet and rewrite the sentence.


### D3. Threshold objective
- **Status:** open
- **Affects:** C6 (threshold function), C8
- **Question:** The proposed rule is **the highest threshold whose recall on PRISM validation is ≥ 0.90**. Recall and
  FPR do not depend on class prevalence, so this choice carries over to the imbalanced real set. Maximising F2, which the
  code does today, does not carry over, because precision depends on prevalence.
- **Default:** recall ≥ 0.90. Decide before any real-data result exists.

### D4. One input size and one intensity normalisation for all models
- **Status:** Done
- **Affects:** C9, C11, C12
- **Question:** Today LensNet sees raw MJy/sr at 300 px, DeepLens uses ZScale at 101 px, TEGLIE a robust clip at
  101 px, and GraViT asinh at 224 px. Downsampling 300 → 101 px gives about 0.089″/px, so a 0.53″ Einstein radius spans
  only about 6 px.
- **Decision:**  300 × 300 px, bands in their natural order 0,1,2,3, and `unit_asinh` (per band, per image, 0.5–99.5
  percentile, asinh softening 0.1) for every model.

### D5. Number of seeds, and whether PRISM is big enough
- **Status:** Done
- **Affects:** C16, E19
- **Question:** 5 seeds per architecture (3 if compute time is short). Also: with only 80 base lens systems, the synthetic
  test set has 12 systems (300 lens images). Should the PRISM authors be asked for more base systems?
- **Default:** 5 seeds, and ask for more systems in parallel.
- **Decision:** 3 seeds, and stay with the 12 systems for now, because of computation limits (all runs are on the
  laptop).

---

## Code changes (`LensDetection` repo)

### C6. PRISM-only training mode
- **Status:** done ([SMesForoush/LensDetection#9](https://github.com/SMesForoush/LensDetection/pull/9)). The grouped split shipped in #7. `--train_data {prism,real,both}` (default
  `prism`, logged as a warning when not given; `--include_prism` = `both`) is in `add_data_arguments`; `--prism_dir`
  defaults to the PRISM data (since 2026-11-21 the laptop copy, before that Roihu's). `system_info.json` and `test_results.json` record `train_data` and
  `split_seed`, and `test_results.json` adds `split_sizes`. `slurm/finetune.sh` takes `TRAIN_DATA` (default `prism`).
  The full PRISM-only LensNet run logged 2,801/600/599 (1,400/300/300 lenses, 56/12/12 systems) and no real images.
- **Depends on:** D1
- **Files:** `datasets/sources.py`, `pipelines/common.py`, `train.py`, `slurm/finetune.sh`, `README.md`
- **Now:** `collect_from_args` always loads the real manifest, and `split_samples` always puts real images in
  train/val/test. `--include_prism` only *adds* PRISM.
- **To do:**
  1. Add a data-mode argument to `add_data_arguments`, for example `--train_data {real,prism,both}`. Its default should
     become `prism`, but changing an existing default affects old scripts, so log it clearly.
  2. With `prism`: do not load the real manifest for training at all, and make `split_samples` return only the PRISM
     splits. `--val_region` and `--test_region` are then unused, so skip their check.
  3. Keep the real manifest loadable on its own for C8: a function that returns all real samples, unsplit.
  4. In `run_finetuning` and `train.py`, record the data mode and the split seed in `system_info.json` and
     `test_results.json`.
  5. Add a `TRAIN_DATA` variable to `slurm/finetune.sh`.
- **Done when:** a 2-epoch smoke run with PRISM only logs train/val/test = 2,801/600/599 (1,400/300/300 lenses) and no
  real images. `system_info.json` shows the mode. Real-data runs still work with `--train_data real`.

### C7. No test-set threshold tuning
- **Status:** done in #7 (`train.py` now uses `best_f2_threshold` on validation).
- **Follow-up:** once D3 is decided, replace `best_f2_threshold` with the agreed rule, for example
  `threshold_for_recall(y_val, s_val, target=0.90)` in `pipelines/common.py`. Use it in both `run_finetuning` and
  `train.py`, and save the threshold **inside `best_model.pth`** (key `threshold`) as well as in `test_results.json`,
  so that C8 can read it.
- **Done when:** the follow-up is merged and a run's checkpoint contains `threshold` and `threshold_rule`.
- **Partly done in C8:** checkpoints already store `threshold` and `threshold_rule` (`THRESHOLD_RULE` in
  `pipelines/common.py`). What is left is the D3 rule itself: swap `best_f2_threshold` and `THRESHOLD_RULE` together.

### C8. Real-data evaluation script
- **Status:** done ([SMesForoush/LensDetection#11](https://github.com/SMesForoush/LensDetection/pull/11)). `pipelines/evaluate_real.py`. `run_finetuning` and
  `train.py` now save the validation threshold in `best_model.pth` (`threshold`, `threshold_rule` =
  `max_f2_on_validation`, a constant in `pipelines/common.py`), so C8 no longer waits for the C7 follow-up. Only the
  rule itself still changes with D3. `train.py` checkpoints also carry `model_name`/`model_config`/`transform`, and
  `load.py` rebuilds older LensNet checkpoints from their `args`, so C8 does not wait for C10 either. Older runs fall back
  to the threshold in their `test_results.json` (only if chosen on validation, and only for the same epoch).
  Smoke-tested on 2-epoch LensNet and DeepLens PRISM runs: 5,031 images scored (277 + 4,754).
- **Depends on:** C6, C7 follow-up
- **Files:** new `scripts/evaluate_real.py` (or `pipelines/evaluate_real.py`), which reuses `pipelines/load.py` and
  `binary_metrics`
- **To do:**
  1. Input: a run directory (or `best_model.pth`), plus the data arguments (`--cosmos_root` and the `--real_*_dir`
     overrides).
     Get the real images with `samples, quality_report = datasets.sources.all_real_samples(args)` (unsplit, no
     `--max_samples_per_class`, corrupted files dropped).
  2. Load the model and its stored transform with `pipelines/load.py::load_finetuned`. LensNet from `train.py` has a
     different checkpoint format, so either handle both or do C10 first.
  3. Read the frozen threshold from the checkpoint. **There must be no threshold argument**, and the script must not
     read validation data.
  4. Score all 5,031 real cutouts. Write `real_predictions.csv` (same columns as `save_predictions`) and
     `real_results.json` with metrics:
     - overall
     - per source (`real_lens_confident`, `real_lens_calibration`, `lookalike`, `real_nonlens`)
     - per lens tier: confident lenses + all non-lenses, and calibration lenses + all non-lenses
  5. Also report `prevalence` (lens fraction) and the random-classifier PR-AUC (= prevalence) next to every PR-AUC.
- **Done when:** running it on a smoke-test checkpoint writes both files, and the counts add up to 277 + 4,754.

### C9. One shared input transform and size
- **Status:** done ([SMesForoush/LensDetection#15](https://github.com/SMesForoush/LensDetection/pull/15)). `--input_preset {thesis,native}` (default `thesis`) in
  `train.py` and every pipeline. `thesis` = `ModelInputTransform.thesis_default()` (`THESIS_TRANSFORM` in
  `datasets/transforms.py`: bands 0,1,2,3, 300 px, `unit_asinh` 0.5–99.5 %, softening 0.1); with it, a `--bands`,
  `--image_size` or `--resize_mode` that differs is an error. `native` = each model's published input (LensNet raw,
  TEGLIE/DeepLens 2,3,1,0 at 101 px, GraViT 3 bands at 224 px; GraViT only supports `native`). `system_info.json`
  records `input_preset` and `transform` (now also from `train.py`); `slurm/finetune.sh` takes `INPUT_PRESET`.
  DeepLens and TEGLIE weights are optional since C11.
- **Depends on:** D4
- **Files:** `datasets/transforms.py`, `pipelines/*.py`, `train.py`
- **Now:** each pipeline builds its own `ModelInputTransform` with a different scaling, size and band order.
  `pipelines/lensnet.py` (C10) already uses the D4 transform as its own default; the legacy `train.py` uses no transform
  (raw MJy/sr; its pilot run collapsed and predicted every image as a lens).
- **To do:** add a single default (for example `ModelInputTransform.thesis_default()`, or defaults in
  `build_arg_parser`) set to the D4 choice, and use it in every from-scratch run. Keep the model-specific settings
  (ZScale, 2,3,1,0 order) reachable only behind an explicit flag for the pretrained side experiment. Record the transform
  config in `system_info.json`, which `run_finetuning` already does.
- **Done when:** the `system_info.json` `transform` entries of all three architectures are identical.

### C10. LensNet in the shared runner, with augmentation
- **Status:** done ([SMesForoush/LensDetection#14](https://github.com/SMesForoush/LensDetection/pull/14)). `pipelines/lensnet.py` trains LensNet from random
  initialisation through `run_finetuning` (no per-model recipe overrides; `--augment` turns on D4 flips/rotations) with
  the D4 input: bands 0,1,2,3, 300 px, `unit_asinh` 0.5–99.5 %, softening 0.1 (`--scaling none` = raw MJy/sr, as
  `train.py`). `load.py` already knew `"lensnet"`, so pipeline checkpoints reload unchanged. `train.py` stays as legacy
  (README and a log warning say so). Grad-CAM moved to `scripts/gradcam.py`, which reads a predictions CSV and any
  checkpoint; its LensNet default layer is now `pool5`, because `conv5` (before BN/ReLU) gave all-zero maps.
  `slurm/finetune.sh` takes `MODEL=lensnet`.
  Smoke run (`--augment --max_samples_per_class 100 --epochs 2`, MPS): about 6 s per epoch after the first, all
  standard outputs written, `load_finetuned` reproduces the saved test scores (max diff 4e-8), and
  `evaluate_real` scores 5,031 real images (277 + 4,754).
- **Depends on:** D2 (if the baseline changes, port that model instead)
- **Files:** new `pipelines/lensnet.py`, `pipelines/load.py`, and possibly the retirement of `train.py`
- **Now:** `train.py` is a separate loop with its own early stopping, metrics, cosine warm restarts and no augmentation.
  This makes LensNet hard to compare fairly with the other models.
- **To do:** add `pipelines/lensnet.py` that builds `models.lensnet.LensNet` and calls `run_finetuning`, as
  `pipelines/teglie.py` does. Add `"lensnet"` to `pipelines/load.py::build_architecture`. Then either leave `train.py`
  as legacy (say so in the README) or delete it. Grad-CAM from `train.py` is worth keeping for E21: move it to a small
  script.
- **Done when:** `python -m pipelines.lensnet --augment ...` runs for 2 epochs and writes the standard outputs, and
  `load_finetuned` reloads it.

### C11. CMU DeepLens from scratch
- **Status:** done ([SMesForoush/LensDetection#17](https://github.com/SMesForoush/LensDetection/pull/17)). `--weights` is optional in `pipelines/deeplens.py` and `pipelines/teglie.py`.
  Without it neither `load_keras_h5` nor `freeze_modules` runs, `--freeze_stages` > 0 is an error, the log says
  `training ... from random initialisation`, and `system_info.json` records `"pretrained": null`. The input is the
  `--input_preset` transform (default `thesis`, C9). `slurm/finetune.sh` takes `PRETRAINED` (default `0` = from
  scratch; `1` adds the published weights). Smoke runs (PRISM, 100 per class, 2 epochs, no weights): DeepLens
  3,557,281 trainable parameters, TEGLIE 3,193,889; both reload with `load_finetuned`. The `--weights` path still
  loads and freezes as before.
- **Files:** `pipelines/deeplens.py`, `slurm/finetune.sh`
- **Now:** `--weights` is `required=True` and `load_keras_h5` is always called, so the model only fine-tunes.
- **To do:** make `--weights` optional. Without it, skip `load_keras_h5` and `freeze_modules`, record
  `"pretrained": null`, and use the shared transform and size (C9) instead of ZScale at 101 px. The dense head depends on
  input size (3×3×512 at 101 px, 9×9×512 at 300 px), and `CMUDeepLens(image_size=...)` already handles that. Do the same
  for `pipelines/teglie.py` if TEGLIE stays in as the optional model.
- **Done when:** `python -m pipelines.deeplens --output_dir ... --max_samples_per_class 100 --epochs 2` runs with no
  weights file and logs about 3.56 M trainable parameters.

### C12. EfficientNet-B0 pipeline
- **Status:** done ([SMesForoush/LensDetection#16](https://github.com/SMesForoush/LensDetection/pull/16)). `pipelines/efficientnet.py` + `models/efficientnet.py`:
  timm `efficientnet_b0`, `pretrained=False`, `in_chans=4`, 2 logits, 4,010,398 parameters; `--dropout` and
  `--drop_path_rate` default to the B0 paper's 0.2. Only `--input_preset thesis` (native is refused); no per-model
  `set_defaults`, so C13 sets its recipe in `build_arg_parser`. `system_info.json` records `"pretrained": null`.
  `load.py` rebuilds `"efficientnet"`. `slurm/finetune.sh` has an `efficientnet` case
  (`MODEL=efficientnet sbatch slurm/finetune.sh --augment`). Smoke test (MPS, `--max_samples_per_class 100 --epochs 2 --augment`): about 12 s/epoch on 137
  training images after the first, train loss 1.70 → 0.92; `load_finetuned` reproduces the saved test probabilities
  to 2e-7.
- **Files:** new `pipelines/efficientnet.py` (optionally `models/efficientnet.py`), `pipelines/load.py`,
  `slurm/finetune.sh`, `README.md`
- **To do:** use `timm.create_model("efficientnet_b0", pretrained=False, in_chans=4, num_classes=2)`, which gives
  4,010,398 parameters. `timm` is already in `requirements-finetune.txt`. `head_params` = the parameters of
  `model.get_classifier()`. Use the shared transform (C9), call `run_finetuning`, and add `"efficientnet"` to
  `build_architecture` in `load.py`. Add a case to `slurm/finetune.sh`.
- **Done when:** a 2-epoch smoke run works and `load_finetuned` reloads the checkpoint.

### C13. One shared training recipe
- **Status:** done ([SMesForoush/LensDetection#19](https://github.com/SMesForoush/LensDetection/pull/19)). The `build_arg_parser` defaults are the thesis recipe: cross-entropy with
  `--class_weighting none` and label smoothing 0, AdamW lr 1e-3, weight decay 1e-4, 1-epoch warm-up + cosine, batch
  64, at most 50 epochs, `--augment` on by default (`--no-augment`), early stopping (patience 10) and checkpoint
  selection on the validation loss (`SELECTION_RULE = "min_val_loss"`); `--min_specificity` is gone. LensNet,
  DeepLens, TEGLIE and EfficientNet set no recipe defaults of their own. Pretrained runs (`--weights`, GraViT) go
  through `parse_args_with_recipe`, which applies `FINETUNE_RECIPE` (the pre-C13 shared values) plus each model's old
  fine-tuning values. `system_info.json` and `test_results.json` record `recipe` and `selection`. Smoke runs
  (MPS, 100 per class, 2 epochs) of LensNet, DeepLens and EfficientNet-B0: `system_info.json` `arguments` differ only
  in model flags (`dropout`, `drop_path_rate`, `freeze_stages`, `weights`, `image_size`, and the input flags that only
  apply with `--input_preset native` or LensNet's `--scaling`), `transform` is identical, and all three reload with
  `load_finetuned`. `slurm/finetune.sh` no longer passes `--min_specificity` (it would not parse).
- **Depends on:** D1, C10
- **Files:** `pipelines/common.py` (`build_arg_parser` defaults, `run_finetuning`), and the per-model
  `p.set_defaults(...)` calls
- **Now:** the recipes differ. Learning rate is 1e-4 everywhere (a fine-tuning value); weight decay is 1e-2 or 1e-4;
  there are class weights and label smoothing 0.1; early stopping is on F2 at 0.5 with a specificity floor; max epochs is
  50 or 100.
- **To do:** make the thesis recipe the default for from-scratch runs:
  - cross-entropy with no class weights (PRISM is 1:1) and label smoothing 0
  - AdamW, lr 1e-3, weight decay 1e-4, 1-epoch warm-up then cosine decay
  - batch 64, at most 50 epochs
  - early stopping with patience 10 on **validation loss**, keeping the lowest-loss checkpoint; remove the
    `min_specificity` gate from checkpoint selection
  - PyTorch default initialisation and augmentation on

  Remove the per-model overrides (`set_defaults(learning_rate=1e-4, ...)`) for from-scratch runs. Any tuning is a
  separate, labelled extension.
- **Done when:** `system_info.json` `arguments` match across the three architectures except for model-specific
  architecture flags.

### C14. Drop corrupted files instead of zero images
- **Status:** done ([SMesForoush/LensDetection#10](https://github.com/SMesForoush/LensDetection/pull/10)). New `datasets/quality.py` checks every file in `collect_from_args`
  and `all_real_samples` (before subsampling), cached per data directory in `file_qc.json`, and drops unreadable,
  wrong-shape, > 50 % NaN/inf and all-zero files. Both runners log `dropped N corrupted files` per source, write
  `data_quality_report.json` and add `dropped_corrupted` to `test_results.json`. `__getitem__` raises instead of
  returning zeros. train.py's old sampled QC report at the end of a run is gone. Local data: 0 of 4,000 PRISM and 0 of
  5,031 real files dropped.
- **Files:** `datasets/base.py` (around lines 231–255 and 275), `datasets/sources.py`, possibly a check script
- **Now:** an image with > 50 % invalid pixels, or one that fails to load, is returned as an all-zero tensor that keeps
  its label (line 275 even returns label 0).
- **To do:** find bad files **before** training (a pass in `collect_from_args`, or a cached `bad_files.json` per data
  directory), drop them from the sample list, and log and save the count per source. Keep the inpainting for files with
  ≤ 50 % invalid pixels. Make `__getitem__` raise instead of returning zeros, so a silent failure can't happen.
- **Done when:** runs log `dropped N corrupted files` per source, and no all-zero tensors reach the model.

### C15. Missing metrics: FPR, bootstrap intervals, curves
- **Status:** done ([SMesForoush/LensDetection#20](https://github.com/SMesForoush/LensDetection/pull/20)). `binary_metrics` returns `fpr` and
  `prevalence` (so every metrics block has them; `evaluate_real.with_baseline` no longer computes its own).
  `bootstrap_ci(y_true, y_score, threshold, n=2000, seed=0)` in `pipelines/common.py` resamples lenses and non-lenses
  separately at the fixed threshold and returns `{metric: [low, high]}` (2.5/97.5 percentiles) for recall,
  specificity, FPR, precision, F1, ROC-AUC and PR-AUC; a metric whose class is missing is `null`. It is stored as
  `test.ci95` in `test_results.json` and as `ci95` in the `overall` and `per_tier` blocks of `real_results.json`; both
  files record the settings under `bootstrap`. `save_curves` writes `roc_curve.csv` (`fpr,tpr,threshold`) and
  `pr_curve.csv` (`recall,precision,threshold`) for the test set, and `evaluate_real` writes `real_roc_curve.csv` /
  `real_pr_curve.csv`. The logged test and real results show the intervals. Legacy `train.py` is unchanged. Smoke run
  (LensNet, 100 per class, 2 epochs, MPS): all files written; the bootstrap adds about 1 s per run on the test split
  and about 8 s to `evaluate_real` (3 blocks of 5,031 images).
- **Files:** `pipelines/common.py::binary_metrics`, `run_finetuning`, and C8's script
- **To do:**
  1. Add `"fpr": fp / (fp + tn)` and `"prevalence": mean(y_true)` to `binary_metrics`.
  2. Add `bootstrap_ci(y_true, y_score, threshold, n=2000, seed=0)`: stratified by class, it returns 95 % percentile
     intervals for recall, specificity, FPR, precision, F1, ROC-AUC and PR-AUC. Store them under `"ci95"` in
     `test_results.json` and `real_results.json`.
  3. Save ROC and PR curve points (`roc_curve.csv`, `pr_curve.csv`) per run, so C16 can overlay seeds. A plotting helper
     can come later.
- **Done when:** `test_results.json` contains `fpr`, `prevalence` and `ci95`, and the curve CSVs exist.

### C16. Multi-seed driver and aggregation
- **Status:** open
- **Depends on:** D5. Most useful after C6, C9–C13.
- **Files:** new `scripts/run_seeds.sh` (a plain sequential loop on the laptop), new `scripts/aggregate_runs.py`
- **To do:**
  1. The driver loops over models × seeds and writes to `results/<model>/seed_<s>/`. It uses one fixed `--split_seed`,
     and only `--seed` varies. Runs go one after another on the laptop's GPU; skip a (model, seed) whose
     `test_results.json` already exists, so the loop can be stopped and resumed. No Slurm.
  2. The aggregator reads every `test_results.json` (and `real_results.json` once C8 exists) and writes a long CSV
     (model, seed, set, metric, value) plus a mean ± sample-sd table per model and set, as CSV and as a LaTeX `tabular`
     snippet for the thesis.
- **Done when:** with two seeds of one smoke-test model, the aggregator prints mean ± sd and writes the `.tex` snippet.

### C17. Error-analysis tools
- **Status:** done ([SMesForoush/LensDetection#13](https://github.com/SMesForoush/LensDetection/pull/13))
- **Depends on:** C8
- **Files:** new `scripts/error_analysis.py`
- **To do** (the procedure is in doc section 5.9):
  1. For each architecture, pick the seed whose real-data ROC-AUC is the median of its seeds.
  2. Extract FP and FN at the frozen threshold from `real_predictions.csv`.
  3. Sample: take all FN. Take all FP if there are ≤ 150; otherwise draw 150 at random with a fixed seed, stratified by
     source (look-alike vs hand-checked).
  4. Render a gallery: a Lupton asinh RGB (R = F444W, G = F277W, B = (F115W + F150W)/2) plus the four single bands per
     object, **with no model score shown** (blind coding).
  5. Write a coding-sheet CSV with columns: object id, source, error type, p(lens) per model, primary category,
     secondary category, notes.
  6. Write a cross-model overlap table (errors made by all models vs errors made by only one).

  Provisional categories:
  - FP: spiral/ring structure; interacting/merging; neighbouring objects; artefacts; ambiguous; other
  - FN: weak/faint arcs; partial/small arcs; poor lens–source contrast; complex foreground; ambiguous/low quality; other
- **Done when:** running it on smoke-test outputs writes the gallery PNGs, an empty coding sheet and the overlap table.

---

## Experiments

### E18. Smoke test every model
- **Status:** open
- **Depends on:** C6, C9–C13
- **To do:** run each architecture locally on MPS for 2 epochs on a small PRISM subset (`--max_samples_per_class 100`).
  Check that the loss goes down, the outputs are written and `load_finetuned` reloads the model. Note the time per
  epoch on the full PRISM set, to estimate how long E19 takes on the laptop.

### E19. Final runs on the laptop
- **Status:** open
- **Depends on:** E18, C16, D5
- **To do:** run 3 architectures × 3 seeds (D5) through `scripts/run_seeds.sh` on the laptop (`--gpu`, MPS), with the
  default data paths. Keep every `system_info.json` and the run directories intact under `results/`. Keep the laptop
  plugged in and awake for the whole run.

### E20. Evaluate every run with frozen thresholds
- **Status:** open
- **Depends on:** E19, C8
- **To do:** evaluate each run on the synthetic test set (done by the training run) and on the real set (C8). Aggregate
  with C16. Report the threshold-free metrics (ROC-AUC, PR-AUC with the prevalence baseline) next to the thresholded
  ones.

### E21. Error analysis
- **Status:** open
- **Depends on:** E20, C17
- **To do:** code the sampled FP and FN with the C17 sheet. Optionally have the supervisor code a random 20 % blind and
  report agreement. Present the results as descriptive counts plus 2–4 example cutouts per major category.

---

## Writing (`../Master-thesis/sections/`)

### W22. Fill the `data.tex` TODOs
- **Status:** open
- **To do:** add the following (open TODOs are at about lines 109, 172, 213, 230–245):
  - PRISM composition: 2,000 lenses = 80 base systems × 25 variations; 2,000 non-lenses = 688 central, 698 pair, 614
    group
  - the grouping key `base_lens_id`
  - the split table: 1,400/300/300 lenses (56/12/12 systems) and 1,401/300/299 non-lenses at split seed 42
  - the real set: 277 lenses (78 confident + 199 calibration) and 4,754 non-lenses (2,608 look-alikes + 2,146
    hand-checked)
  - the cutout format: 300 × 300 px at 0.03″/px, MJy/sr

  Remove the outdated "~400 lenses / ~1,000 non-lenses" numbers.

### W23. Synthetic vs real statistics for the limitations section
- **Status:** open
- **To do:** recompute the per-band background (median) and noise (1.4826 × MAD) on the **full** PRISM and real sets.
  A spot check of 60 cutouts each found:
  - PRISM noise in F277W/F444W is about 2.5× the real value (0.023/0.025 vs 0.009/0.011)
  - PRISM backgrounds are 2–4× higher (0.0036–0.0066 vs about 0.0015 MJy/sr)

  Write a small script (`scripts/domain_stats.py`) that outputs a CSV and a LaTeX table, and ask the PRISM authors
  whether the offset is a sky pedestal.

### W24. Write `methodology.tex`
- **Status:** open
- **Depends on:** D1–D5 and the final code
- **To do:** turn the Claude Doc "Chapter 5 — Methods" (https://claude.ai/artifact/DhvJQrpNLMjFPxvQKbd9mA, sections
  5.1–5.9) into LaTeX. Take the hyperparameters and software versions from the final runs' `system_info.json`, not from
  the doc. Update the `literature.tex` baseline sentence according to D2.

### W25. Pilot and pretrained runs
- **Status:** open
- **To do:** keep the real-data pilot runs in `results/` (LensNet ROC-AUC 0.53, collapsed; DeepLens 0.56; TEGLIE 0.62;
  GraViT unfinished) out of the main results. Mention them at most as a pilot that motivates synthetic training (168
  real training lenses are not enough). Label any pretrained fine-tuning or GraViT run as a side experiment.

---

## Optional (future)

### O26. Bring the Roihu scripts up to date
- **Status:** optional, not planned. Do it only if a CSC Roihu budget becomes available and we decide to run there.
- **Where it stands:** `slurm/finetune.sh` has a case for all five models (`MODEL=`, `PRETRAINED=`, `TRAIN_DATA=`,
  `INPUT_PRESET=`) as of C12 and passes `--prism_dir`/`--cosmos_root` explicitly, which it must, because the defaults
  now point to the laptop. Nothing has run on Roihu since the pilot. The multi-seed driver (C16) is a laptop loop with
  no Slurm array. `submit_training.sh` (untracked) is an older Mahti job for `train.py`, and `train.py`'s
  `setup_mahti_environment` still describes Mahti.
- **Files:** `slurm/finetune.sh`, `scripts/run_seeds.sh`, `README.md`, possibly `train.py`, `submit_training.sh`
- **To do:**
  1. Check that `module load python-pytorch/<version>` still exists on Roihu and has `timm`, and that every change
     made after C12 is reflected in `slurm/finetune.sh`.
  2. Give `scripts/run_seeds.sh` a Slurm job-array mode (`--array=0-<models × seeds - 1>`, index → (model, seed)),
     next to the laptop loop.
  3. Check that the Roihu data paths (`ROIHU_PRISM_DIR`, `ROIHU_COSMOS_ROOT` in `datasets/sources.py`) still exist and
     hold the same files as the laptop copies (same PRISM files, same manifest), so laptop and Roihu runs are
     comparable.
  4. Mixed precision is CUDA-only, so Roihu (GH200) runs would differ from laptop (MPS, float32) runs. Do not mix the
     two in one results table; rerun every seed on one machine.
- **Done when:** a 2-epoch smoke job of each architecture runs on Roihu and writes the standard outputs.
