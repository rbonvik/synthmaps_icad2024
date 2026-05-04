# %%
# profile_metrics.py
#
# Profile the simulation metrics that will drive the FM-synth sonification.
# Generates a set of paper-ready figures and a numerical summary CSV.
#
# Outputs (under <synthmapspath>/figures/profiling/):
#   - 01_distributions.png         per-metric distribution (linear + log)
#   - 02_per_dataset_ranges.png    boxplot per dataset per metric
#   - 03_timeseries_examples.png   trajectories for a few representative datasets
#   - 04_rate_distributions.png    distributions of first differences (rate of change)
#   - 05_correlation.png           cross-metric correlation heatmap + pairplot
#   - 06_coverage_3d.png           where data lives in joint metric space
#   - summary.csv                  per (dataset, metric) statistics

# %%
# imports
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from utils import get_path

# %%
# config
PARAMS = ["dipolar_energy", "magnet_flips", "total_mag_angle", "ice_rule_density"]
PARAM_LABELS = {
    "dipolar_energy": "Dipolar energy",
    "magnet_flips": "Magnet flips",
    "total_mag_angle": "Total magnetic angle",
    "ice_rule_density": "Ice rule density",
}
N_TIMESERIES_EXAMPLES = 6   # how many datasets to plot in the trace grid
DPI = 200
FIGSIZE_WIDE = (14, 6)


# %%
# load selected datasets manifest and resolve metric CSV paths
def load_manifest():
    """Return (datasets_list, metrics_root) where metrics_root is the
    directory the relative metrics_csv paths in the manifest are relative to."""
    selected_path = get_path("selecteddatasetspath")
    metrics_root = os.path.dirname(get_path("metricspath"))  # parent of metrics_out/
    with open(selected_path, "r") as f:
        manifest = json.load(f)
    return manifest["datasets"], metrics_root


def load_all_metrics(datasets, metrics_root):
    """Load every dataset's metrics CSV into a dict keyed by dataset name.
    Each value is a DataFrame with the columns from the CSV plus a 'dataset' col."""
    out = {}
    for d in datasets:
        path = os.path.join(metrics_root, d["metrics_csv"])
        if not os.path.exists(path):
            print(f"  [warn] missing: {path}")
            continue
        df = pd.read_csv(path)
        df["dataset"] = d["name"]
        out[d["name"]] = df
    return out


# %%
# numerical summary
def build_summary(all_metrics):
    """Per-(dataset, metric) descriptive stats. Saved as summary.csv."""
    rows = []
    for name, df in all_metrics.items():
        for col in PARAMS:
            v = df[col].values
            rows.append({
                "dataset": name,
                "metric": col,
                "n": len(v),
                "min": np.min(v),
                "max": np.max(v),
                "mean": np.mean(v),
                "median": np.median(v),
                "std": np.std(v),
                "pct_zero": np.mean(v == 0) * 100,
                "range": np.max(v) - np.min(v),
            })
    return pd.DataFrame(rows)


# %%
# plot 1 — per-metric distribution overview (pooled across datasets)
def plot_distributions(all_metrics, out_path):
    """Two-row grid: linear-scale histogram (top) and log/symlog (bottom)
    for each of the four metrics. Pooled across all datasets."""
    fig, axes = plt.subplots(2, len(PARAMS), figsize=(5 * len(PARAMS), 8), dpi=DPI)
    pooled = {p: np.concatenate([df[p].values for df in all_metrics.values()])
              for p in PARAMS}

    for j, p in enumerate(PARAMS):
        v = pooled[p]
        # row 0 — linear scale
        ax = axes[0, j]
        ax.hist(v, bins=80, color="steelblue", alpha=0.85)
        ax.set_title(PARAM_LABELS[p])
        ax.set_ylabel("count" if j == 0 else "")
        ax.grid(alpha=0.3)

        # row 1 — log y to make rare values visible
        ax = axes[1, j]
        ax.hist(v, bins=80, color="steelblue", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(PARAM_LABELS[p])
        ax.set_ylabel("count (log)" if j == 0 else "")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Metric distributions across all selected datasets", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# plot 2 — per-dataset ranges (boxplot per dataset per metric)
def plot_per_dataset_ranges(all_metrics, out_path):
    """Boxplot showing each dataset's distribution per metric.
    Helps decide whether per-dataset normalization is warranted."""
    names = list(all_metrics.keys())
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(max(10, 0.5 * len(names)), 9), dpi=DPI)

    for i, p in enumerate(PARAMS):
        ax = axes[i]
        data = [all_metrics[n][p].values for n in names]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="lightsteelblue", alpha=0.7))
        ax.set_ylabel(PARAM_LABELS[p])
        ax.grid(alpha=0.3, axis="y")
        if i < len(PARAMS) - 1:
            ax.set_xticklabels([])
        else:
            # truncate names so they fit
            short = [n.split("_", 1)[0] for n in names]
            ax.set_xticklabels(short, rotation=90, fontsize=7)
            ax.set_xlabel("dataset")

    fig.suptitle("Per-dataset distribution of each metric (outliers hidden)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# plot 3 — temporal traces for representative datasets
def plot_timeseries_examples(all_metrics, out_path, n=N_TIMESERIES_EXAMPLES):
    """Small multiples: rows = datasets, cols = metrics. Shows raw temporal
    evolution so reviewers can see whether things are smooth, spiky, etc."""
    names = list(all_metrics.keys())
    # pick spaced-out indices so we sample across the manifest
    if len(names) <= n:
        chosen = names
    else:
        idx = np.linspace(0, len(names) - 1, n).astype(int)
        chosen = [names[i] for i in idx]

    fig, axes = plt.subplots(len(chosen), len(PARAMS),
                             figsize=(4 * len(PARAMS), 1.8 * len(chosen)),
                             dpi=DPI, sharex="col")

    # ensure axes is 2D even if one row
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)

    for r, name in enumerate(chosen):
        df = all_metrics[name]
        for c, p in enumerate(PARAMS):
            ax = axes[r, c]
            ax.plot(df["time"].values, df[p].values, lw=0.7, color="steelblue")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(PARAM_LABELS[p])
            if c == 0:
                # short label on left
                short = name.split("_", 1)[0] + "\n" + name.split("_", 1)[1][:18]
                ax.set_ylabel(short, fontsize=8, rotation=0,
                              ha="right", va="center")
            if r == len(chosen) - 1:
                ax.set_xlabel("time step")

    fig.suptitle("Temporal evolution of each metric (sampled datasets)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# plot 4 — rate-of-change distributions
def plot_rate_distributions(all_metrics, out_path):
    """Distributions of first differences. Reveals whether the metric carries
    most of its information in changes (rate-coded) vs. levels."""
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(5 * len(PARAMS), 4), dpi=DPI)

    for j, p in enumerate(PARAMS):
        diffs = np.concatenate([np.diff(df[p].values) for df in all_metrics.values()])
        ax = axes[j]
        # use symlog x so we can see negative diffs and zeros at once
        v = diffs
        # clip extreme percentiles for plotting only
        lo, hi = np.percentile(v, [0.5, 99.5])
        v_clipped = v[(v >= lo) & (v <= hi)]
        ax.hist(v_clipped, bins=80, color="indianred", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"Δ {PARAM_LABELS[p]}")
        ax.set_xlabel("change between consecutive steps")
        if j == 0:
            ax.set_ylabel("count (log)")
        ax.grid(alpha=0.3, which="both")
        # annotate with % zeros
        pct_zero = np.mean(diffs == 0) * 100
        ax.text(0.02, 0.95, f"% zero Δ: {pct_zero:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    fig.suptitle("Rate-of-change distributions (5–99.5 percentile clipped for display)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# plot 5 — cross-metric correlation
def plot_correlation(all_metrics, out_path):
    """Two-panel figure: left = correlation heatmap pooled across all datasets,
    right = pairwise scatter matrix on a downsample."""
    pooled = pd.concat([df[PARAMS] for df in all_metrics.values()], ignore_index=True)

    # downsample for scatter (pooled can be 100ks of points)
    n_total = len(pooled)
    n_sample = min(20000, n_total)
    sampled = pooled.sample(n=n_sample, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI,
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # left — correlation heatmap
    corr = pooled.corr()
    im = axes[0].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(PARAMS)))
    axes[0].set_yticks(range(len(PARAMS)))
    axes[0].set_xticklabels([PARAM_LABELS[p] for p in PARAMS], rotation=30, ha="right")
    axes[0].set_yticklabels([PARAM_LABELS[p] for p in PARAMS])
    for i in range(len(PARAMS)):
        for k in range(len(PARAMS)):
            axes[0].text(k, i, f"{corr.values[i, k]:.2f}",
                         ha="center", va="center",
                         color="white" if abs(corr.values[i, k]) > 0.5 else "black")
    axes[0].set_title("Pearson correlation (pooled)")
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    # right — 2D hexbin between metric 0 and metric 2 (most informative usually)
    ax = axes[1]
    hb = ax.hexbin(sampled[PARAMS[0]], sampled[PARAMS[2]],
                   gridsize=50, cmap="viridis", mincnt=1, bins="log")
    ax.set_xlabel(PARAM_LABELS[PARAMS[0]])
    ax.set_ylabel(PARAM_LABELS[PARAMS[2]])
    ax.set_title(f"Joint density: {PARAMS[0]} vs {PARAMS[2]}")
    plt.colorbar(hb, ax=ax, label="log(count)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# plot 6 — 3D coverage of joint metric space
def plot_coverage_3d(all_metrics, out_path):
    """3D scatter showing where the data trajectories live in the joint
    (energy, monopole_count, magnet_flips) space. Points colored by dataset."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.tab20
    names = list(all_metrics.keys())
    n_max_per_ds = 2000  # avoid millions of points

    for i, name in enumerate(names):
        df = all_metrics[name]
        if len(df) > n_max_per_ds:
            df = df.sample(n=n_max_per_ds, random_state=42)
        ax.scatter(df[PARAMS[0]], df[PARAMS[1]], df[PARAMS[2]],
                   s=2, alpha=0.4, color=cmap(i % 20), label=name.split("_", 1)[0])

    ax.set_xlabel(PARAM_LABELS[PARAMS[0]])
    ax.set_ylabel(PARAM_LABELS[PARAMS[1]])
    ax.set_zlabel(PARAM_LABELS[PARAMS[2]])
    ax.set_title("Coverage of joint metric space (downsampled)")

    # legend would be huge — show a small one
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.05, 1.0),
              ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
# main
def main():
    synthmaps_root = get_path("synthmapspath")
    out_dir = os.path.join(synthmaps_root, "figures", "profiling")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")

    print("Loading manifest...")
    datasets, metrics_root = load_manifest()
    print(f"  manifest declares {len(datasets)} datasets, "
          f"resolving relative paths against {metrics_root}")

    print("Loading metric CSVs...")
    all_metrics = load_all_metrics(datasets, metrics_root)
    print(f"  loaded {len(all_metrics)} datasets")
    if len(all_metrics) == 0:
        raise RuntimeError("no datasets loaded — check paths.json and the manifest")

    print("Building summary table...")
    summary = build_summary(all_metrics)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary.groupby("metric")[["min", "max", "mean", "pct_zero"]].agg(["min", "max"]))

    print("Plot 1 — distributions...")
    plot_distributions(all_metrics, os.path.join(out_dir, "01_distributions.png"))

    print("Plot 2 — per-dataset ranges...")
    plot_per_dataset_ranges(all_metrics, os.path.join(out_dir, "02_per_dataset_ranges.png"))

    print("Plot 3 — timeseries examples...")
    plot_timeseries_examples(all_metrics, os.path.join(out_dir, "03_timeseries_examples.png"))

    print("Plot 4 — rate distributions...")
    plot_rate_distributions(all_metrics, os.path.join(out_dir, "04_rate_distributions.png"))

    print("Plot 5 — correlation...")
    plot_correlation(all_metrics, os.path.join(out_dir, "05_correlation.png"))

    print("Plot 6 — 3D coverage...")
    plot_coverage_3d(all_metrics, os.path.join(out_dir, "06_coverage_3d.png"))

    print("Done.")
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()