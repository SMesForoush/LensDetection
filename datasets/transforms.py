"""Model-specific input transforms applied on top of LensDatasetMemoryEfficient.

The base dataset yields raw (4, 300, 300) float tensors. Each pretrained model expects a different
input convention, so each fine-tuning pipeline wraps the base dataset with one of these transforms:

    GraViT   : 3 bands -> 224x224 -> per-band [0, 1] stretch -> ImageNet mean/std
    TEGLIE   : 4 bands (r, i, g, u slots) -> 101x101 -> clipped & scaled to [-100, 100]
    DeepLens : 4 bands (R, I, G, U slots) -> 101x101 -> per-band astropy ZScale to [0, 1]

`bands` maps the model's input channels to channels of the base data, e.g. bands=(2, 3, 1, 0) feeds base
channel 2 into the model's first slot. All transforms are pure torch/numpy and run in DataLoader workers.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_bands(text):
    return tuple(int(b) for b in str(text).split(","))


def resize_image(x, size, mode="resize"):
    """(C, H, W) -> (C, size, size) by antialiased bilinear resampling or centre cropping."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    if mode == "center_crop":
        h, w = x.shape[-2:]
        if size > min(h, w):
            raise ValueError(f"Cannot center-crop {h}x{w} to {size}x{size}")
        top, left = (h - size) // 2, (w - size) // 2
        return x[:, top:top + size, left:left + size]
    if mode == "resize":
        return F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
    raise ValueError(f"Unknown resize mode '{mode}'")


def _band_quantiles(x, q):
    return torch.quantile(x.flatten(1), torch.tensor(q, dtype=x.dtype), dim=1).T  # (C, len(q))


def unit_stretch(x, low_pct=0.5, high_pct=99.5, stretch="asinh", asinh_softening=0.1):
    """Per-band percentile normalisation to [0, 1], optionally followed by an asinh stretch."""
    qs = _band_quantiles(x, [low_pct / 100.0, high_pct / 100.0])
    lo, hi = qs[:, :1, None], qs[:, 1:, None]
    x = ((x - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    if stretch == "asinh":
        x = torch.asinh(x / asinh_softening) / np.arcsinh(1.0 / asinh_softening)
    elif stretch != "linear":
        raise ValueError(f"Unknown stretch '{stretch}'")
    return x


def teglie_scaling(x, mode="robust", clip_pct=99.5, flux_scale=1.0):
    """TEGLIE clipping/scaling to [-100, 100].

    mode='absolute' reproduces TEGLIE preprocessing.scaling_clipping exactly (clip at +-1e-9 in KiDS flux
    units) after multiplying the data by `flux_scale` to convert units. mode='robust' replaces the absolute
    1e-9 limit with each band's `clip_pct` percentile, which keeps the same output range for data in other
    units (e.g. JWST MJy/sr).
    """
    if mode == "absolute":
        vmax = 1e-9
        x = x * flux_scale
        return x.clamp(-vmax, vmax) / vmax * 100.0
    if mode == "robust":
        vmax = _band_quantiles(x, [clip_pct / 100.0])[:, :, None].clamp_min(1e-12)
        return torch.max(torch.min(x, vmax), -vmax) / vmax * 100.0
    raise ValueError(f"Unknown TEGLIE scaling mode '{mode}'")


def zscale_bands(x, contrast=0.25):
    """Per-band astropy ZScaleInterval normalisation to [0, 1], as in LEXACTUM utils/load_fits.py."""
    from astropy.visualization import ZScaleInterval

    interval = ZScaleInterval(contrast=contrast)
    out = np.stack([interval(band, clip=True) for band in x.numpy()])
    return torch.from_numpy(out.astype(np.float32))


class ModelInputTransform:
    """Band selection -> resize -> intensity scaling (-> ImageNet normalisation)."""

    def __init__(self, bands, image_size, resize_mode="resize", scaling="unit_asinh", imagenet_norm=False,
                 augment=False, **scaling_kwargs):
        self.bands = tuple(bands)
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.scaling = scaling
        self.imagenet_norm = imagenet_norm
        self.augment = augment
        self.scaling_kwargs = scaling_kwargs

    def config(self):
        return dict(bands=list(self.bands), image_size=self.image_size, resize_mode=self.resize_mode,
                    scaling=self.scaling, imagenet_norm=self.imagenet_norm, **self.scaling_kwargs)

    def without_augmentation(self):
        return ModelInputTransform(augment=False, **self.config())

    def __call__(self, x):
        x = resize_image(x[list(self.bands)], self.image_size, self.resize_mode)
        if self.scaling == "unit_asinh":
            x = unit_stretch(x, stretch="asinh", **self.scaling_kwargs)
        elif self.scaling == "unit_linear":
            x = unit_stretch(x, stretch="linear", **self.scaling_kwargs)
        elif self.scaling == "teglie":
            x = teglie_scaling(x, **self.scaling_kwargs)
        elif self.scaling == "zscale":
            x = zscale_bands(x, **self.scaling_kwargs)
        elif self.scaling != "none":
            raise ValueError(f"Unknown scaling '{self.scaling}'")
        if self.imagenet_norm:
            mean = torch.tensor(IMAGENET_MEAN[:x.shape[0]], dtype=x.dtype)[:, None, None]
            std = torch.tensor(IMAGENET_STD[:x.shape[0]], dtype=x.dtype)[:, None, None]
            x = (x - mean) / std
        if self.augment:
            # Lensing is invariant to the dihedral group: random flips + 90 degree rotations
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(-2, -1))
        return x.contiguous()


class TransformedDataset(Dataset):
    """Applies a ModelInputTransform to every sample of a base dataset."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transform(x), y
