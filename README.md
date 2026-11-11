# LensDetection

## Data and training from scratch

`train.py` trains LensNet from scratch, and `pipelines/` fine-tunes pretrained models. Both read the same
data sources through `datasets/sources.py` and use the same split.

**Real COSMOS-Web DR1 images (the default).** `datasets/cosmos_real_manifest.csv` lists them, with their sky region:

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

**Split by sky region.** DR1 is divided into 5 sky regions. Region `--test_region` (default 0) is the test set,
`--val_region` (default 4) the validation set, and the other three regions are training data. Training-region
images within 13" of the validation or test region are dropped, so no training cutout shares pixels with a
validation or test cutout.

**PRISM simulations (optional).** `--include_prism --prism_dir DIR` adds `cosmos_lens_*.npy` (`prism_lens`, label 1)
and `cosmos_nonlens_*.npy` (`prism_nonlens`, label 0). They have no sky position and are split 70/20/10 at random
(stratified), so validation and test then hold both real and PRISM images. `--mock_data_dir` is accepted as
another name for `--prism_dir`.

```bash
python train.py --output_dir results/lensnet_real --gpu
python train.py --include_prism --prism_dir /scratch/ituomine/COSMOS-WEB-LENS/prism_jwst_cosmosweb_v5 \
    --output_dir results/lensnet_real_prism --gpu
INCLUDE_PRISM=1 sbatch slurm/finetune.sh       # fine-tuning on real + PRISM data
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
python train.py --cosmos_root cosmos_data --real_lens_dir mock-lens-modeling/real_cowls_lenses --output_dir results/local
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

sbatch slurm/finetune.sh gravit   --repo_id parlange/vit-gravit-c3 --weights weights/gravit/vit-gravit-c3/model.safetensors
sbatch slurm/finetune.sh teglie   --freeze_stages 1
sbatch slurm/finetune.sh deeplens
```

Or run directly, e.g. `python -m pipelines.teglie --help`. Every pipeline accepts the `train.py` data
arguments (`--cosmos_root`, `--val_region`, `--test_region`, `--include_prism`, `--prism_dir`,
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
`test_results.json`, `test_predictions.csv`, `validation_predictions.csv`, `system_info.json`, `finetune.log`.
The decision threshold is tuned for F2 on the validation set and then applied to the test set.

Reload a fine-tuned model together with its exact input transform:

```python
from pipelines.load import load_finetuned, lens_probability
model, transform = load_finetuned("best_model.pth")
p_lens = lens_probability(model, transform, raw_images)   # raw_images: (N, 4, 300, 300)
```

### Notes on the pretrained models

- GraViT: `config.json` names the wrong architecture for pit, mlp-mixer and twins_*. CvT is a HuggingFace
  `transformers` CvT-21, not a timm model. `models/gravit.py` maps each family to the architecture that
  matches the published weights exactly. The ResNet-18 baselines ship only pickles and are not supported.
- TEGLIE was trained on KiDS-like fluxes and clips at +-1e-9. `--scaling_mode robust` (default) clips each band
  at its 99.5th percentile instead; `--scaling_mode absolute --flux_scale <k>` reproduces the original.
- CMU DeepLens' dense head depends on the input size. With `--image_size` other than 101, only the
  convolutional weights are reused.
