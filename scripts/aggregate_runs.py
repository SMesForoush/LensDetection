#!/usr/bin/env python3
"""Aggregate multi-seed runs into mean +- sample sd per model and evaluation set (TODO C16).

    python scripts/aggregate_runs.py results                           # the scripts/run_seeds.sh layout
    python scripts/aggregate_runs.py results/smoke --output_dir results/smoke/aggregate

Input: run directories, or folders searched recursively for them. A run directory holds the test_results.json of a
training run (pipelines/common.py::run_finetuning) and, once pipelines/evaluate_real.py has scored it, a
real_results.json. The model and seed come from the run's system_info.json (else real_results.json, else the
<model>/seed_<s> directory names of scripts/run_seeds.sh).

Evaluation sets (one row block each):
    prism_test        the synthetic PRISM test split, at the threshold chosen on PRISM validation (test_results.json "test";
                      only runs trained with --train_data prism, whose test split is PRISM)
    real              every real COSMOS-Web image, at the same frozen threshold (real_results.json "overall")
    real_confident    confident real lenses + all real non-lenses (real_results.json "per_tier")
    real_calibration  calibration real lenses + all real non-lenses
Metrics: the threshold-free ROC-AUC and PR-AUC, the PR-AUC of a random classifier (= prevalence), and recall,
specificity, FPR, precision, F1, F2 and accuracy at the frozen threshold, plus the threshold and the best epoch.

Writes to --output_dir (default <first input>/aggregate):
    runs.csv          one row per run: directory, model, seed, split seed, train data, which result files it has
    metrics_long.csv  model, seed, set, metric, value (one row per run x set x metric)
    summary.csv       model, set, metric, n_seeds, mean, sd (sample sd, ddof = 1), min, max, seeds
    summary_<set>.tex one LaTeX tabular per set (booktabs; rows = models, cells = mean $\\pm$ sd), for \\input in a
                      table environment of the thesis
Prints the mean +- sd table. Runs that do not share one --split_seed, or that did not train on PRISM only, are
reported, because their seeds do not form one comparable group.
"""
import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("aggregate_runs")

SETS = ("prism_test", "real", "real_confident", "real_calibration")
SET_LABELS = {"prism_test": "PRISM test (synthetic)", "real": "COSMOS-Web, all real images",
              "real_confident": "COSMOS-Web, confident lenses + all non-lenses",
              "real_calibration": "COSMOS-Web, calibration lenses + all non-lenses"}
METRICS = ("roc_auc", "pr_auc", "pr_auc_random", "recall", "specificity", "fpr", "precision", "f1", "f2", "acc",
           "threshold", "best_epoch")
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "pr_auc_random": "PR-AUC (random)", "recall": "Recall",
                 "specificity": "Specificity", "fpr": "FPR", "precision": "Precision", "f1": "F$_1$",
                 "f2": "F$_2$", "acc": "Accuracy (\\%)", "threshold": "Threshold", "best_epoch": "Best epoch"}
TEX_METRICS = ("roc_auc", "pr_auc", "recall", "specificity", "fpr", "precision", "f1")
MODEL_ORDER = ("lensnet", "deeplens", "efficientnet", "teglie", "gravit")
MODEL_LABELS = {"lensnet": "LensNet", "deeplens": "CMU DeepLens", "efficientnet": "EfficientNet-B0",
                "teglie": "TEGLIE", "gravit": "GraViT"}


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("runs", nargs="*", default=["results"],
                   help="Run directories, or folders searched recursively for test_results.json / real_results.json")
    p.add_argument("--output_dir", help="Where to write the tables (default: <first input>/aggregate)")
    p.add_argument("--models", nargs="+", help="Only these models (default: every model found)")
    p.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS), help="Evaluation sets to aggregate")
    p.add_argument("--tex_metrics", nargs="+", choices=METRICS, default=list(TEX_METRICS),
                   help="Columns of the LaTeX tables")
    p.add_argument("--decimals", type=int, default=3, help="Decimals in the printed and LaTeX tables")
    return p.parse_args(argv)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def find_run_dirs(inputs):
    """Every directory under the inputs that holds a test_results.json or a real_results.json."""
    dirs = set()
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_file():
            path = path.parent
        if (path / "test_results.json").exists() or (path / "real_results.json").exists():
            dirs.add(path.resolve())
        else:
            dirs.update(f.parent.resolve() for name in ("test_results.json", "real_results.json")
                        for f in path.rglob(name))
    return sorted(dirs)


def seed_from_path(run_dir):
    m = re.fullmatch(r"seed_(\d+)", run_dir.name)
    return int(m.group(1)) if m else None


def with_derived(m):
    """A metrics block plus FPR and prevalence (from the confusion matrix) where the block does not have them yet."""
    m = dict(m)
    cm = m.get("confusion_matrix")
    if cm:
        if "fpr" not in m:
            m["fpr"] = cm["FP"] / (cm["FP"] + cm["TN"]) if cm["FP"] + cm["TN"] else float("nan")
        n = cm["TN"] + cm["FP"] + cm["FN"] + cm["TP"]
        if "prevalence" not in m and n:
            m["prevalence"] = (cm["TP"] + cm["FN"]) / n
    if "pr_auc_random" not in m and "prevalence" in m:
        m["pr_auc_random"] = m["prevalence"]  # the PR-AUC of a random classifier
    return m


def read_run(run_dir):
    """(run info, {set: metrics block}) of one run directory, or None if it holds nothing usable."""
    system = read_json(run_dir / "system_info.json") if (run_dir / "system_info.json").exists() else {}
    test = read_json(run_dir / "test_results.json") if (run_dir / "test_results.json").exists() else None
    real = read_json(run_dir / "real_results.json") if (run_dir / "real_results.json").exists() else None
    arguments = system.get("arguments", {})

    model = system.get("model_name") or (real or {}).get("model_name") or \
        (run_dir.parent.name if seed_from_path(run_dir) is not None else None)
    seed = arguments.get("seed", (real or {}).get("seed"))
    if seed is None:
        seed = seed_from_path(run_dir)
    path_seed = seed_from_path(run_dir)
    if path_seed is not None and seed is not None and int(seed) != path_seed:
        logger.warning(f"{run_dir}: trained with --seed {seed} but its directory says seed {path_seed}; using {seed}")
    if model is None or seed is None:
        logger.warning(f"{run_dir}: cannot tell the model or seed (no system_info.json, no <model>/seed_<s> path); "
                       f"skipped")
        return None

    # As pipelines/evaluate_real.py: runs from before --train_data existed trained on real data (the old default).
    train_data = (test or {}).get("train_data") or system.get("train_data") or arguments.get("train_data") or \
        (real or {}).get("train_data")
    if train_data is None and (system or test is not None):
        train_data = "both" if arguments.get("include_prism") else "real"

    blocks = {}
    if test is not None:
        if "test" not in test:
            logger.warning(f"{run_dir}/test_results.json has no 'test' block (a legacy train.py run); "
                           f"its test metrics are skipped")
        elif train_data != "prism":
            logger.warning(f"{run_dir}: trained with --train_data {train_data}, so its test split is not the PRISM "
                           f"test split; its test metrics are skipped")
        else:
            blocks["prism_test"] = {**with_derived(test["test"]), "best_epoch": test.get("best_epoch")}
    if real is not None:
        blocks["real"] = with_derived(real["overall"])
        for tier, m in real.get("per_tier", {}).items():
            blocks[f"real_{tier}"] = with_derived(m)
    info = {"run_dir": str(run_dir), "model": model, "seed": int(seed),
            "split_seed": (test or {}).get("split_seed", system.get("split_seed", arguments.get("split_seed"))),
            "train_data": train_data,
            "recipe": (test or {}).get("recipe", system.get("recipe")),
            "has_test_results": test is not None, "has_real_results": real is not None}
    return info, blocks


def model_sort_key(model):
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model)


def collect(run_dirs, models=None, sets=SETS):
    runs, rows = [], []
    for run_dir in run_dirs:
        read = read_run(run_dir)
        if read is None:
            continue
        info, blocks = read
        if models and info["model"] not in models:
            continue
        runs.append(info)
        for set_name in sets:
            for metric in METRICS:
                value = blocks.get(set_name, {}).get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    rows.append({"model": info["model"], "seed": info["seed"], "set": set_name, "metric": metric,
                                 "value": float(value)})
    return pd.DataFrame(runs), pd.DataFrame(rows, columns=["model", "seed", "set", "metric", "value"])


def check_runs(runs):
    """Warn about runs that cannot be averaged together as seeds of one experiment."""
    dupes = runs[runs.duplicated(["model", "seed"], keep=False)]
    for (model, seed), g in dupes.groupby(["model", "seed"]):
        logger.warning(f"{model} seed {seed} appears {len(g)} times ({', '.join(g['run_dir'])}); all are averaged. "
                       f"Pass only one results folder per experiment.")
    split_seeds = runs["split_seed"].dropna().unique()
    if len(split_seeds) > 1:
        logger.warning(f"The runs use different --split_seed values {sorted(split_seeds)}; "
                       f"only --seed should vary between repetitions")
    not_prism = runs[runs["train_data"].notna() & (runs["train_data"] != "prism")]
    for _, r in not_prism.iterrows():
        logger.warning(f"{r['run_dir']}: trained with --train_data {r['train_data']}, not PRISM only; "
                       f"its real-set metrics are not an external evaluation")


def summarise(long):
    def seeds(s):
        return " ".join(str(x) for x in sorted(s))
    g = long.groupby(["model", "set", "metric"])
    summary = g["value"].agg(n_seeds="count", mean="mean", sd=lambda v: v.std(ddof=1), min="min", max="max")
    summary["seeds"] = g["seed"].agg(seeds)
    summary = summary.reset_index()
    summary["_m"] = summary["model"].map(model_sort_key)
    summary["_s"] = summary["set"].map(SETS.index)
    summary["_k"] = summary["metric"].map(METRICS.index)
    return summary.sort_values(["_s", "_m", "_k"]).drop(columns=["_m", "_s", "_k"]).reset_index(drop=True)


def mean_sd(mean, sd, n, decimals, tex=False):
    pm = " $\\pm$ " if tex else " ± "
    if n < 2 or not math.isfinite(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}{pm}{sd:.{decimals}f}"


def metric_decimals(metric, decimals):
    return 0 if metric == "best_epoch" else (1 if metric == "acc" else decimals)


def print_summary(summary, decimals):
    for set_name, s in summary.groupby("set", sort=False):
        print(f"\n{SET_LABELS[set_name]} ({set_name})")
        metrics = [m for m in METRICS if m in set(s["metric"])]
        models = sorted(s["model"].unique(), key=model_sort_key)
        out = pd.DataFrame("-", index=models, columns=["seeds", *metrics], dtype=object)
        for model, g in s.groupby("model"):
            out.loc[model, "seeds"] = int(g["n_seeds"].max())
        for _, r in s.iterrows():
            out.loc[r["model"], r["metric"]] = mean_sd(r["mean"], r["sd"], int(r["n_seeds"]),
                                                       metric_decimals(r["metric"], decimals))
        print(out.to_string())


def tex_table(summary_set, set_name, metrics, decimals):
    s = summary_set.set_index(["model", "metric"])
    models = sorted(summary_set["model"].unique(), key=model_sort_key)
    n_seeds = summary_set.groupby("model")["n_seeds"].max()
    lines = [f"% {SET_LABELS[set_name]}: mean $\\pm$ sample sd over seeds (scripts/aggregate_runs.py)"]
    prevalence = summary_set[summary_set["metric"] == "pr_auc_random"]["mean"]
    if len(prevalence):
        lines.append(f"% Prevalence (lens fraction) = PR-AUC of a random classifier: {prevalence.iloc[0]:.4f}")
    lines += [f"\\begin{{tabular}}{{l{'c' * (len(metrics) + 1)}}}", "\\toprule",
              " & ".join(["Model", "Seeds", *(METRIC_LABELS[m] for m in metrics)]) + " \\\\", "\\midrule"]
    for model in models:
        cells = [MODEL_LABELS.get(model, model.replace("_", "\\_")), str(int(n_seeds[model]))]
        for metric in metrics:
            if (model, metric) in s.index:
                r = s.loc[(model, metric)]
                cells.append(mean_sd(r["mean"], r["sd"], int(r["n_seeds"]), metric_decimals(metric, decimals), tex=True))
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_arguments(argv)
    run_dirs = find_run_dirs(args.runs)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.runs[0]) / "aggregate"
    run_dirs = [d for d in run_dirs if output_dir.resolve() not in (d, *d.parents)]
    if not run_dirs:
        logger.error(f"No test_results.json or real_results.json under {' '.join(args.runs)}")
        return 1

    runs, long = collect(run_dirs, args.models, args.sets)
    if runs.empty or long.empty:
        logger.error("No usable runs found" + (f" for models {args.models}" if args.models else ""))
        return 1
    runs = runs.sort_values(["model", "seed"], key=lambda c: c.map(model_sort_key) if c.name == "model" else c)
    check_runs(runs)
    for model, g in runs.groupby("model", sort=False):
        logger.info(f"{model}: {len(g)} run(s), seeds {' '.join(str(s) for s in g['seed'])}"
                    + ("" if g["has_real_results"].all() else
                       f" ({(~g['has_real_results']).sum()} without real_results.json)"))
    if (runs.groupby("model").size() < 2).any():
        logger.warning("Some models have a single seed; their sd is undefined and shown as the mean only")

    summary = summarise(long)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "runs.csv", index=False)
    long.to_csv(output_dir / "metrics_long.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    tex_files = []
    for set_name, s in summary.groupby("set", sort=False):
        path = output_dir / f"summary_{set_name}.tex"
        path.write_text(tex_table(s, set_name, args.tex_metrics, args.decimals))
        tex_files.append(path.name)

    print_summary(summary, args.decimals)
    print()
    logger.info(f"Wrote runs.csv, metrics_long.csv, summary.csv and {', '.join(tex_files)} to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
