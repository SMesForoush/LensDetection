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

**MIRAGE-v1.1 simulations (optional).** `--include_mirage` adds the images of `datasets/mirage_v1_1_manifest.csv`:
real DR1 galaxies (hosts) with a simulated lensed arc added (`mirage_lens`, label 1), the same host with the same
source galaxy placed next to it without lensing (`mirage_companion`, label 0), and the host alone (`mirage_host`,
label 0). Each image has the sky region of its host, so MIRAGE is split by region together with the real images and
validation and test then hold both. MIRAGE hosts lie at least 13" from every COWLS lens and no PSF of the 79
confident lenses was used, so no simulated image contains a COWLS lens. (The PRISM simulations are not used: they are
made from the COWLS lenses themselves, test lenses included, and have no sky position.)

The MIRAGE manifest is built on Roihu by `scripts/build_mirage_manifest.py` from
`datasets/mirage_v1_1/training_rows.csv`. It stops if an image is missing or if a host's sky region differs from
the region table of the real manifest. It leaves out host images that are already in the real manifest (a look-alike
or hand-checked non-lens there) and the host and companion images of AnomalyMatch candidates.

```bash
python train.py --output_dir results/lensnet_real --gpu
python scripts/build_mirage_manifest.py --project_dir /scratch/ituomine/COSMOS-WEB-LENS/mock-lens-modeling   # once
python train.py --include_mirage --output_dir results/lensnet_real_mirage --gpu
INCLUDE_MIRAGE=1 sbatch slurm/finetune.sh      # fine-tuning on real + MIRAGE data
```

**Running off Roihu.** `scripts/pack_for_laptop.py` zips the manifests' images on Roihu (Python 3 standard library
only). By default it packs the real non-lenses, about 6.9 GB uncompressed. `--with_lenses` also packs the lens
cutouts, `--with_mirage` the MIRAGE images (roughly 50 GB; `--mirage_hosts N` keeps those of N random host galaxies)
and `--weights_dir` the pretrained weights. The script checks that every file exists and that the zip will fit
before it writes anything.

```bash
# on Roihu, from the repository
python3 scripts/pack_for_laptop.py --with_mirage --mirage_hosts 2000 --weights_dir weights \
    --out /scratch/$USER/cosmos_laptop_data.zip
# on the laptop
rsync -P roihu:/scratch/$USER/cosmos_laptop_data.zip . && unzip cosmos_laptop_data.zip -d cosmos_data
python train.py --cosmos_root cosmos_data --real_lens_dir mock-lens-modeling/real_cowls_lenses \
    --include_mirage --mirage_manifest cosmos_data/mirage_v1_1_manifest.csv --output_dir results/local --gpu
```

`--gpu` uses an NVIDIA GPU (CUDA) when there is one, and otherwise the Apple Silicon GPU (MPS) on a Mac, in
`train.py` and all three pipelines. Mixed precision stays CUDA-only. On an M-series Mac, CMU DeepLens epochs ran
about 20x faster on the GPU than on the CPU.

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
arguments (`--cosmos_root`, `--val_region`, `--test_region`, `--include_mirage`,
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
