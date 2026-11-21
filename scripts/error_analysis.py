#!/usr/bin/env python3
"""Structured error analysis of the real-data evaluation (TODO C17, thesis section "Structured error analysis").

    python scripts/error_analysis.py results --output_dir results/error_analysis   # data paths default to the laptop

Input: run directories scored by pipelines/evaluate_real.py, i.e. holding real_predictions.csv and real_results.json.
Each positional argument is such a directory, or a folder searched recursively for them (e.g. results/ with the
results/<model>/seed_<s>/ layout). Runs are grouped into architectures by the `model_name` in real_results.json.

Procedure (fixed before any real-data error is looked at):
  1. Choose the runs. Per architecture, the run whose real-data ROC-AUC is the median over its seeds. With an even
     number of seeds, the lower of the two middle runs (ties broken by seed). Only the chosen runs are used below.
  2. Extract the errors at the frozen threshold, i.e. the `outcome` column of real_predictions.csv: false positives
     (FP, a non-lens scored as a lens) and false negatives (FN, a lens that was missed).
  3. Sample. All FN. All FP if there are at most --max_fp (150); otherwise --max_fp of them, drawn with --sample_seed
     and stratified by source (lookalike / real_nonlens), proportional to each source's share of the FP.
  4. Render a gallery: for every sampled image, an asinh RGB composite (Lupton et al. 2004; R = F444W, G = F277W,
     B = (F115W + F150W) / 2) plus the four single bands. No model score or model name is shown (blind coding).
  5. Write an empty coding sheet: one row per sampled image (the union over architectures).
  6. Write a cross-model overlap table over ALL errors of the chosen runs (not only the sample): errors made by every
     architecture vs by only one.

Writes to --output_dir:
    selected_runs.csv      every run found, its real-data ROC-AUC and whether it was chosen
    sample.csv             (model, object_id, source, error_type): which images each architecture's sample holds,
                           for the per-architecture counts
    coding_sheet.csv       object_id, source, error_type, p_lens_<model> for every architecture, primary_category,
                           secondary_category, notes (empty), plus sampled_by and the gallery file. The scores are there
                           for the analysis; hide those columns while coding, so the coding stays blind to the models.
    error_categories.csv   the provisional categories (thesis table "Provisional error categories")
    gallery/<FP|FN>/<object_id>.png
    error_overlap.csv      one row per image that any chosen run got wrong: which architectures got it wrong
    overlap_table.csv      per error type: distinct errors, made by all architectures, by only <model>, by k of N
    error_analysis.json    the settings, chosen runs and counts
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.sources import add_real_data_arguments, real_dirs, real_samples  # noqa: E402

BANDS = ("F115W", "F150W", "F277W", "F444W")  # channels 0-3
ARCSEC_PER_PX = 0.03
ERROR_TYPES = ("FN", "FP")
CATEGORIES = [  # (error type, category, typical appearance): methodology.tex, tab:method-error-categories
    ("FP", "spiral/ring structure", "Spiral arms, star-forming rings or tidal shells curving around a bulge"),
    ("FP", "interacting/merging", "Tidal tails or bridges, or a close pair that mimics a counter-image"),
    ("FP", "neighbouring objects", "Unrelated companions or line-of-sight galaxies at arc-like positions"),
    ("FP", "artefacts", "Diffraction spikes, bright-star halos, detector or mosaic-edge features"),
    ("FP", "ambiguous", "A plausible lens candidate; the label may be wrong"),
    ("FN", "weak/faint arcs", "Arc barely above the noise"),
    ("FN", "partial/small arcs", "Short arc or small Einstein radius, close to or inside the lens light"),
    ("FN", "poor lens-source contrast", "Source colour or brightness close to that of the lens galaxy"),
    ("FN", "complex foreground", "Disc or merging lens galaxy, or a crowded field"),
    ("FN", "ambiguous/low quality", "Low-score candidate, noisy or hard to read"),
    ("both", "other", "Fits no category; described in free text"),
]

logger = logging.getLogger("error_analysis")


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("runs", nargs="+", help="Run directories with real_predictions.csv + real_results.json, "
                                           "or folders to search for them")
    p.add_argument("--output_dir", default="results/error_analysis")
    p.add_argument("--max_fp", type=int, default=150, help="Most false positives sampled per architecture")
    p.add_argument("--sample_seed", type=int, default=42, help="Seed of the false-positive sample")
    g = p.add_argument_group("gallery")
    g.add_argument("--skip_gallery", action="store_true", help="Write the tables only (no image files needed)")
    g.add_argument("--crop", type=int, default=0, help="Show only the central N x N pixels (0: the whole cutout)")
    g.add_argument("--softening", type=float, default=2.0,
                   help="The asinh stretch is linear up to this many noise sigma (per image), logarithmic above")
    g.add_argument("--white_percentile", type=float, default=99.9,
                   help="Per-image intensity percentile that is shown as white")
    add_real_data_arguments(p.add_argument_group("real data (to find the image files for the gallery)"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# 1. Runs
# --------------------------------------------------------------------------------------
def find_runs(paths):
    """Directories holding both real_results.json and real_predictions.csv, under or at each path."""
    runs = set()
    for path in map(Path, paths):
        if not path.exists():
            raise FileNotFoundError(path)
        if (path / "real_results.json").is_file():
            candidates = [path]
        else:
            candidates = [r.parent for r in path.rglob("real_results.json")]
        for run in candidates:
            if (run / "real_predictions.csv").is_file():
                runs.add(run.resolve())
            else:
                logger.warning(f"{run}: real_results.json without real_predictions.csv, skipped")
    if not runs:
        raise FileNotFoundError(f"No real_results.json + real_predictions.csv under {paths} "
                                f"(run pipelines/evaluate_real.py first)")
    return sorted(runs)


def read_run(run):
    with open(run / "real_results.json") as f:
        r = json.load(f)
    o = r["overall"]
    return {"model": r["model_name"], "seed": r.get("seed"), "roc_auc": o["roc_auc"], "pr_auc": o["pr_auc"],
            "recall": o["recall"], "fpr": o["fpr"], "threshold": r["threshold"],
            "threshold_rule": r.get("threshold_rule"),
            "train_data": r.get("train_data"), "external_evaluation": r.get("external_evaluation"), "run": str(run)}


def choose_median_runs(runs):
    """All runs with a `chosen` column: per model, the (lower) median run by real-data ROC-AUC."""
    df = pd.DataFrame([read_run(r) for r in runs])
    df["seed_sort"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.sort_values(["model", "roc_auc", "seed_sort", "run"]).reset_index(drop=True)
    df["chosen"] = False
    for model, g in df.groupby("model", sort=True):
        df.loc[g.index[(len(g) - 1) // 2], "chosen"] = True
        if g["seed"].isna().any() or g["seed"].duplicated().any():
            logger.warning(f"{model}: seeds {g['seed'].tolist()} are missing or repeated; are these really one "
                           f"architecture's seed repetitions? Runs: {g['run'].tolist()}")
        for col in ("train_data", "threshold_rule"):
            if g[col].nunique(dropna=False) > 1:
                logger.warning(f"{model}: runs differ in {col} ({g[col].tolist()})")
        if not g["external_evaluation"].fillna(False).all():
            logger.warning(f"{model}: some runs were not trained on PRISM only, so their real-data errors are not "
                           f"an external evaluation")
    return df.drop(columns="seed_sort")


# --------------------------------------------------------------------------------------
# 2. Errors
# --------------------------------------------------------------------------------------
def manifest_ids(manifest):
    """{file name: object id} from the real manifest (file names are unique there)."""
    T = pd.read_csv(manifest, dtype={"id": str}, keep_default_na=False)
    names = T["file"].map(lambda f: Path(f).name)
    if names.duplicated().any():
        raise ValueError(f"{manifest}: file names are not unique, cannot map predictions to object ids")
    return dict(zip(names, T["id"]))


def read_predictions(run, threshold, ids):
    """real_predictions.csv with an object_id column, checked against the frozen threshold."""
    P = pd.read_csv(Path(run) / "real_predictions.csv")
    P["object_id"] = P["file_path"].map(lambda f: ids.get(Path(f).name))
    if P["object_id"].isna().any():
        raise ValueError(f"{run}: {int(P['object_id'].isna().sum())} predictions not in the manifest, "
                         f"e.g. {P.loc[P['object_id'].isna(), 'file_path'].iloc[0]}")
    if P["object_id"].duplicated().any():
        raise ValueError(f"{run}: an object is scored more than once")
    predicted = (P["lens_probability"] >= threshold).astype(int)
    clear = (P["lens_probability"] - threshold).abs() > 1e-6  # ignore float round-off at the threshold itself
    if (predicted != P["predicted_label"])[clear].any():
        raise ValueError(f"{run}: predicted_label does not match lens_probability >= {threshold}")
    return P.rename(columns={"data_source": "source"})[
        ["object_id", "source", "true_label", "lens_probability", "outcome", "file_path"]]


# --------------------------------------------------------------------------------------
# 3. Sample
# --------------------------------------------------------------------------------------
def allocate(counts, n):
    """Split n over strata in proportion to counts (largest remainder; ties to the alphabetically first stratum)."""
    total = sum(counts.values())
    exact = {k: n * c / total for k, c in counts.items()}
    out = {k: int(np.floor(v)) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def sample_errors(errors, max_fp, seed):
    """All FN; all FP if at most max_fp, else a source-stratified random sample of max_fp."""
    fn = errors[errors["outcome"] == "FN"]
    fp = errors[errors["outcome"] == "FP"].sort_values("object_id")
    if len(fp) > max_fp:
        rng = np.random.default_rng(seed)
        quota = allocate(fp["source"].value_counts().to_dict(), max_fp)
        fp = pd.concat([g.iloc[np.sort(rng.choice(len(g), quota[s], replace=False))]
                        for s, g in sorted(fp.groupby("source"))])
    return pd.concat([fn, fp]).sort_values(["outcome", "object_id"])


# --------------------------------------------------------------------------------------
# 4. Gallery
# --------------------------------------------------------------------------------------
def robust_sigma(x):
    """Noise level: 1.4826 x the median absolute deviation."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def asinh_scale(x, soft, white):
    """Asinh stretch (Lupton et al. 2004): linear below `soft`, logarithmic above, `white` -> 1, negatives -> 0."""
    return np.clip(np.arcsinh(np.clip(x, 0, None) / soft) / np.arcsinh(white / soft), 0, 1)


def stretch_limits(x, softening, percentile):
    """(soft, white) for a sky-subtracted image: soft = softening x noise sigma, white = the given percentile."""
    soft = max(softening * robust_sigma(x), 1e-30)
    return soft, max(np.percentile(x, percentile), 2 * soft)


def lupton_rgb(r, g, b, softening, percentile):
    """Colour-preserving asinh RGB (Lupton et al. 2004): every channel is scaled by f(I)/I, I = mean of the three.
    The sky (median) is subtracted per channel; pixels whose brightest channel exceeds 1 are scaled down as a
    whole, so bright regions keep their colour."""
    rgb = np.stack([c - np.median(c) for c in (r, g, b)], axis=-1)
    intensity = rgb.mean(axis=-1)
    soft, white = stretch_limits(intensity, softening, percentile)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(intensity > 0, np.arcsinh(intensity / soft) / np.arcsinh(white / soft) / intensity, 0.0)
    rgb = np.clip(rgb * factor[..., None], 0, None)
    return rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1.0)


def band_image(x, softening, percentile):
    x = x - np.median(x)
    return asinh_scale(x, *stretch_limits(x, softening, percentile))


def load_cutout(path, crop):
    x = np.nan_to_num(np.load(path).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 3 or x.shape[0] != len(BANDS):
        raise ValueError(f"{path}: expected shape (4, H, W), got {x.shape}")
    if crop and crop < min(x.shape[1:]):
        y0, x0 = (x.shape[1] - crop) // 2, (x.shape[2] - crop) // 2
        x = x[:, y0:y0 + crop, x0:x0 + crop]
    return x


def render(path, out_png, title, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = load_cutout(path, args.crop)
    panels = [("RGB: F444W / F277W / (F115W+F150W)/2",
               lupton_rgb(x[3], x[2], (x[0] + x[1]) / 2, args.softening, args.white_percentile))]
    panels += [(band, band_image(x[i], args.softening, args.white_percentile)) for i, band in enumerate(BANDS)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.35))
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, origin="lower", cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    h, w = x.shape[1:]
    bar = 1 / ARCSEC_PER_PX  # 1 arcsec
    axes[0].plot([0.05 * w, 0.05 * w + bar], [0.06 * h] * 2, color="white", lw=2)
    axes[0].text(0.05 * w + bar / 2, 0.09 * h, '1"', color="white", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def image_paths(args):
    """{object id: local file} for every real image, resolved like pipelines/evaluate_real.py does."""
    T = pd.read_csv(args.real_manifest, dtype={"id": str}, keep_default_na=False)
    samples = real_samples(args.real_manifest, args.cosmos_root, real_dirs(args))
    return dict(zip(T["id"], (s.path for s in samples)))


# --------------------------------------------------------------------------------------
# 6. Overlap
# --------------------------------------------------------------------------------------
def overlap_table(overlap, models):
    rows = []
    n = len(models)
    for error_type in ERROR_TYPES:
        E = overlap[overlap["error_type"] == error_type]
        rows.append((error_type, "distinct errors", len(E)))
        rows.append((error_type, f"made by all {n} models", int((E["n_models_wrong"] == n).sum())))
        for k in range(n - 1, 1, -1):
            rows.append((error_type, f"made by {k} of {n} models", int((E["n_models_wrong"] == k).sum())))
        for m in models:
            rows.append((error_type, f"made only by {m}", int(((E["n_models_wrong"] == 1) & E[f"wrong_{m}"]).sum())))
        for m in models:
            rows.append((error_type, f"made by {m} (total)", int(E[f"wrong_{m}"].sum())))
    return pd.DataFrame(rows, columns=["error_type", "group", "count"])


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. One run per architecture
    runs = choose_median_runs(find_runs(args.runs))
    runs.to_csv(out / "selected_runs.csv", index=False)
    chosen = runs[runs["chosen"]].set_index("model")
    models = list(chosen.index)
    for model, g in runs.groupby("model"):
        c = chosen.loc[model]
        logger.info(f"{model}: {len(g)} run(s), ROC-AUC {sorted(round(v, 4) for v in g['roc_auc'])}; "
                    f"chosen seed {c['seed']} (ROC-AUC {c['roc_auc']:.4f}): {c['run']}")

    # 2. Errors at the frozen threshold, 3. sample
    ids = manifest_ids(args.real_manifest)
    preds = {m: read_predictions(chosen.loc[m, "run"], chosen.loc[m, "threshold"], ids) for m in models}
    samples, counts = [], {}
    for m in models:
        errors = preds[m][preds[m]["outcome"].isin(ERROR_TYPES)]
        sample = sample_errors(errors, args.max_fp, args.sample_seed)
        samples.append(sample.assign(model=m))
        counts[m] = {t: {"errors": errors[errors["outcome"] == t]["source"].value_counts().to_dict(),
                         "sampled": sample[sample["outcome"] == t]["source"].value_counts().to_dict()}
                     for t in ERROR_TYPES}
        logger.info(f"{m}: {len(errors)} errors ({counts[m]['FN']['errors']} FN, {counts[m]['FP']['errors']} FP); "
                    f"sampled {len(sample)}")
    sample = pd.concat(samples).rename(columns={"outcome": "error_type"})
    sample[["model", "object_id", "source", "error_type"]].to_csv(out / "sample.csv", index=False)

    # Scores of every model for every image
    scores = pd.concat({m: p.set_index("object_id")["lens_probability"] for m, p in preds.items()}, axis=1)
    scores.columns = [f"p_lens_{m}" for m in scores.columns]
    meta = pd.concat([p.set_index("object_id")[["source", "true_label"]] for p in preds.values()])
    meta = meta[~meta.index.duplicated()]

    # 5. Coding sheet: one row per sampled image
    sheet = (sample.groupby("object_id")
             .agg(source=("source", "first"), error_type=("error_type", "first"),
                  sampled_by=("model", lambda s: ";".join(sorted(s))))
             .join(scores.round(4)).reset_index().sort_values(["error_type", "source", "object_id"]))
    sheet["primary_category"] = sheet["secondary_category"] = sheet["notes"] = ""
    sheet["gallery_png"] = [f"gallery/{t}/{i}.png" for t, i in zip(sheet["error_type"], sheet["object_id"])]
    sheet = sheet[["object_id", "source", "error_type", *scores.columns, "primary_category", "secondary_category",
                   "notes", "sampled_by", "gallery_png"]]
    sheet.to_csv(out / "coding_sheet.csv", index=False)
    pd.DataFrame(CATEGORIES, columns=["error_type", "category", "typical_appearance"]).to_csv(
        out / "error_categories.csv", index=False)

    # 6. Overlap over all errors of the chosen runs
    wrong = pd.concat({m: p.set_index("object_id")["outcome"].isin(ERROR_TYPES) for m, p in preds.items()}, axis=1)
    wrong = wrong.eq(True)  # an image a model did not score counts as not wrong
    wrong = wrong[wrong.any(axis=1)]
    overlap = meta.loc[wrong.index].assign(error_type=np.where(meta.loc[wrong.index, "true_label"] == 1, "FN", "FP"))
    overlap["n_models_wrong"] = wrong.sum(axis=1)
    overlap["wrong_by"] = [";".join(m for m in models if r[m]) for r in wrong.to_dict("records")]
    for m in models:
        overlap[f"wrong_{m}"] = wrong[m]
    overlap = overlap.join(scores).reset_index().sort_values(["error_type", "n_models_wrong", "object_id"],
                                                             ascending=[True, False, True])
    overlap.drop(columns="true_label").to_csv(out / "error_overlap.csv", index=False)
    table = overlap_table(overlap, models)
    table.to_csv(out / "overlap_table.csv", index=False)
    logger.info("Cross-model overlap (all errors of the chosen runs):\n" + table.to_string(index=False))

    # 4. Gallery
    rendered = 0
    if args.skip_gallery:
        logger.info("--skip_gallery: no images rendered")
    else:
        files = image_paths(args)
        for i, row in enumerate(sheet.itertuples(index=False), 1):
            render(files[row.object_id], out / row.gallery_png, f"{row.object_id}  ({row.error_type})", args)
            rendered += 1
            if i % 50 == 0 or i == len(sheet):
                logger.info(f"Gallery: {i} / {len(sheet)}")

    summary = {
        "inputs": [str(p) for p in args.runs], "output_dir": str(out),
        "settings": {"max_fp": args.max_fp, "sample_seed": args.sample_seed, "crop": args.crop,
                     "softening": args.softening,
                     "white_percentile": args.white_percentile, "real_manifest": str(args.real_manifest)},
        "selection_rule": "per model, the run with the median real-data ROC-AUC (lower median for an even count)",
        "chosen_runs": chosen.reset_index()[["model", "seed", "roc_auc", "threshold", "run"]].to_dict("records"),
        "counts": counts,
        "coding_sheet_rows": int(len(sheet)), "gallery_pngs": rendered,
        "overlap": {t: dict(zip(g["group"], g["count"].astype(int))) for t, g in table.groupby("error_type")},
    }
    with open(out / "error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Wrote {len(sheet)} coding-sheet rows, {rendered} gallery PNGs and the overlap table to {out}")
    return summary


if __name__ == "__main__":
    main()
