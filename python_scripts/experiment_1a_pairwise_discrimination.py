"""
experiment_1b_physics_regression.py
====================================
Experiment 1B: Recover physics metric values from audio features.

Question
--------
If a mapping faithfully encodes a physics metric in audio, then a
regressor trained on audio features should be able to predict that
metric's values on unseen datasets. If mapping A puts dipolar_energy
on pitch (the most audible parameter), audio features should recover
dipolar_energy well in mapping A. In mapping A_swap, where
magnet_flips drives pitch instead, magnet_flips should be the
most recoverable.

Method
------
For each (mapping, feature_space, physics_metric):
  • Split the 80 datasets into K folds (default 5).
  • For each fold:
      - Train a regressor on per-timestep (audio_features → metric_value)
        pairs from the K-1 training folds (~64 datasets).
      - Predict per-timestep metric values on the held-out fold (~16
        datasets, none seen during training).
      - Score with R² (coefficient of determination).
  • Report mean R² across folds.

R² interpretation:
  1.0   — perfect recovery of metric values from audio
  0.0   — no better than predicting the mean (no signal)
  < 0   — worse than the mean (model fails to generalise; common with
          poor mappings or feature spaces that carry no relevant signal)

Why leave-DATASETS-out, not leave-timesteps-out?
  Random timestep splits leak: timesteps from each dataset appear in
  both train and test, so the model can memorise per-dataset audio
  prototypes rather than learning a generalisable audio → physics
  relationship. We want generalisation to *new* physical trajectories,
  so we hold out whole datasets.

Predicted pattern if mappings preserve their intended metric
------------------------------------------------------------
  A & B:                   dipolar_energy   recovers best (it drives pitch)
  A_swap & A_rotate:       magnet_flips     recovers best (it drives pitch)
  hamming_from_init:       moderate in A/B (drives harm_ratio); weaker
                           in A_swap (where it drives harm_ratio still
                           — a wash on this metric); strongest in
                           A_rotate (drives harm_ratio).

If the predicted pattern holds, mappings DO preserve intended physics
information in audio — just in ways the correlational RSA framework
couldn't detect. That would be a clean positive result.

If the pattern does NOT hold (e.g., the same metric recovers best in
every mapping, or none recover well), the negative finding is stronger.

Outputs (under <synthmapspath>/figures/evaluation/)
---------------------------------------------------
    physics_regression_r2.png        — heatmap of mean R² per
                                        (mapping, feature_space) for
                                        each physics metric.
    experiment_1b_results.csv        — per-(mapping, feature_space,
                                        physics_metric, fold) R².

Usage
-----
    python experiment_1b_physics_regression.py
    python experiment_1b_physics_regression.py --regressor rf
    python experiment_1b_physics_regression.py --n_folds 5 --n_jobs 16
    python experiment_1b_physics_regression.py --max_train_rows 20000
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_path, frequency2midi

# ── config ────────────────────────────────────────────────────────────────────

MAPPINGS = ["A", "B", "C", "D", "E", "F"]

PHYSICS_METRICS = ["dipolar_energy", "magnet_flips", "hamming_from_init"]

PERCEPTUAL_COLS = [
    "hardness", "depth", "brightness", "roughness", "warmth",
    "sharpness", "boominess",
]
SPECTRAL_COLS = [
    "spectral_centroid", "spectral_crest", "spectral_decrease",
    "spectral_energy", "spectral_flatness", "spectral_kurtosis",
    "spectral_roll_off", "spectral_skewness", "spectral_slope",
    "spectral_spread", "inharmonicity",
]

FEATURE_LABELS = {
    "pitch":      "Pitch only",
    "fm_params":  "FM params",
    "perceptual": "Perceptual",
    "spectral":   "Spectral",
    "mel":        "Mel spec.",
    "encodec":    "EnCodec",
    "clap":       "CLAP",
}
FEATURE_ORDER = ["pitch", "fm_params", "perceptual", "spectral", "mel", "encodec", "clap"]

# Which physics metric drives the *pitch* slot in each mapping. Used to
# annotate the figure with the prediction "metric driving pitch should
# recover best."
PITCH_METRIC = {
    "A":         "dipolar_energy",
    "B":         "dipolar_energy",
    "A_swap":    "magnet_flips",
    "A_rotate":  "magnet_flips",
}

DPI = 200


# ── path helpers ──────────────────────────────────────────────────────────────

def get_mapping_paths(mapping: str) -> dict:
    root = get_path("synthmapspath")
    out  = os.path.join(root, "results", f"mapping_{mapping}")
    return {
        "params_csv":     os.path.join(root, "mapped_params", f"mapping_{mapping}",
                                       "_all_datasets.csv"),
        "perceptual_csv": os.path.join(out, "fm_synth_perceptual_features.csv"),
        "spectral_csv":   os.path.join(out, "fm_synth_spectral_features.csv"),
        "mel_npy":        os.path.join(out, "fm_synth_mel_spectrograms_mean.npy"),
        "encodec_npy":    os.path.join(out, "fm_synth_encodec_embeddings.npy"),
        "clap_npy":       os.path.join(out, "fm_synth_clap_embeddings.npy"),
    }


def get_output_dir() -> str:
    d = os.path.join(get_path("synthmapspath"), "figures", "evaluation")
    os.makedirs(d, exist_ok=True)
    return d


# ── data loading ──────────────────────────────────────────────────────────────

def _clean(X: np.ndarray) -> np.ndarray:
    """Replace inf/NaN with column medians. Standardisation happens
    later inside each fold to avoid leaking test statistics."""
    X = X.astype(np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isinf(col)] = np.nan
        if np.all(np.isnan(col)):
            col = np.zeros_like(col)
        else:
            col = np.where(np.isnan(col), np.nanmedian(col), col)
        X[:, j] = col
    return X


def _resolve_metrics_csv(metrics_csv: str, metrics_root: str, name: str) -> str | None:
    """Try several path resolutions for a manifest's metrics_csv field.

    The manifest has historically stored this field in different ways:
    bare filename, relative path, or absolute path. We try the obvious
    candidates and return the first that exists, or None if none do.
    """
    candidates = [
        metrics_csv,                                      # absolute path or already-resolved
        os.path.join(metrics_root, metrics_csv),          # bare filename under metrics_root
        os.path.join(metrics_root, os.path.basename(metrics_csv)),  # subdir-prefixed → strip and re-join
        os.path.join(metrics_root, f"{name}.csv"),        # last resort: derive from dataset name
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_all_physics(metrics_root: str, manifest_path: str) -> pd.DataFrame:
    """Load all physics CSVs into a single DataFrame keyed by (dataset, time)."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    frames = []
    n_missing = 0
    for meta in manifest["datasets"]:
        path = _resolve_metrics_csv(meta["metrics_csv"], metrics_root, meta["name"])
        if path is None:
            print(f"  [warn] missing physics CSV for {meta['name']} "
                  f"(tried under {metrics_root})")
            n_missing += 1
            continue
        df = pd.read_csv(path)
        df["dataset"] = meta["name"]
        keep = ["dataset", "time"] + [m for m in PHYSICS_METRICS if m in df.columns]
        frames.append(df[keep])
    if not frames:
        raise RuntimeError(
            f"No physics CSVs found. Checked {len(manifest['datasets'])} "
            f"manifest entries under '{metrics_root}'. Verify metricspath "
            f"in paths.json points at the directory containing the CSVs."
        )
    if n_missing:
        print(f"  [warn] {n_missing} of {len(manifest['datasets'])} physics CSVs "
              f"could not be located")
    return pd.concat(frames, ignore_index=True)


def load_mapping_data(mapping: str, all_physics: pd.DataFrame) -> dict | None:
    """Load audio features and merge per-row physics values for one mapping."""
    paths = get_mapping_paths(mapping)
    if not os.path.exists(paths["params_csv"]):
        print(f"  [skip] params CSV missing for {mapping}")
        return None

    params_df = pd.read_csv(paths["params_csv"]).reset_index(drop=True)
    N = len(params_df)
    if "dataset" not in params_df.columns or "time" not in params_df.columns:
        raise ValueError(f"params CSV for {mapping} needs 'dataset' and 'time'")

    # Merge in physics values, row-aligned with params_df.
    phys_join = params_df[["dataset", "time"]].merge(
        all_physics, on=["dataset", "time"], how="left",
    )
    physics: dict[str, np.ndarray] = {}
    for m in PHYSICS_METRICS:
        if m in phys_join.columns:
            physics[m] = phys_join[m].values.astype(np.float64)

    features: dict[str, np.ndarray] = {}

    midi = frequency2midi(params_df["freq"].values.astype(np.float64))
    features["pitch"] = _clean(midi.reshape(-1, 1))

    X_fm = np.column_stack([
        midi,
        params_df["harm_ratio"].values,
        params_df["mod_index"].values,
    ])
    features["fm_params"] = _clean(X_fm)

    if os.path.exists(paths["perceptual_csv"]):
        df = pd.read_csv(paths["perceptual_csv"], index_col=0)
        avail = [c for c in PERCEPTUAL_COLS if c in df.columns]
        features["perceptual"] = _clean(df[avail].reindex(range(N)).values)

    if os.path.exists(paths["spectral_csv"]):
        df = pd.read_csv(paths["spectral_csv"], index_col=0)
        avail = [c for c in SPECTRAL_COLS if c in df.columns]
        features["spectral"] = _clean(df[avail].reindex(range(N)).values)

    if os.path.exists(paths["mel_npy"]):
        mels = np.load(paths["mel_npy"])[:N]
        features["mel"] = _clean(mels)

    if os.path.exists(paths["encodec_npy"]):
        embs = np.load(paths["encodec_npy"])[:N]
        features["encodec"] = _clean(embs.reshape(len(embs), -1))

    if os.path.exists(paths["clap_npy"]):
        embs = np.load(paths["clap_npy"])[:N]
        features["clap"] = _clean(embs)

    return {
        "labels":   params_df["dataset"].values,
        "features": features,
        "physics":  physics,
    }


# ── regression ────────────────────────────────────────────────────────────────

def make_regressor(kind: str):
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    if kind == "rf":
        return RandomForestRegressor(n_estimators=50, max_depth=12,
                                      n_jobs=1, random_state=42)
    raise ValueError(f"unknown regressor: {kind}")


def run_one_fold(args):
    """Train on rows whose dataset is in `train_datasets`, test on
    rows from `test_datasets`. Returns (mapping, feature_space, metric,
    fold, r2)."""
    (mapping, feat_name, metric_name,
     X, y, labels, train_datasets, test_datasets, fold_idx,
     regressor_kind, max_train_rows, seed) = args

    train_mask = np.isin(labels, train_datasets)
    test_mask  = np.isin(labels, test_datasets)

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask],  y[test_mask]

    valid_tr = np.isfinite(ytr)
    valid_te = np.isfinite(yte)
    Xtr, ytr = Xtr[valid_tr], ytr[valid_tr]
    Xte, yte = Xte[valid_te], yte[valid_te]

    if len(Xtr) < 10 or len(Xte) < 10:
        return {
            "mapping": mapping, "feature_space": feat_name,
            "physics_metric": metric_name, "fold": fold_idx,
            "r2": np.nan, "n_train": len(Xtr), "n_test": len(Xte),
        }

    if max_train_rows is not None and len(Xtr) > max_train_rows:
        rng = np.random.default_rng(seed + fold_idx)
        idx = rng.choice(len(Xtr), max_train_rows, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    reg = make_regressor(regressor_kind)
    reg.fit(Xtr_s, ytr)
    yhat = reg.predict(Xte_s)
    r2 = float(r2_score(yte, yhat))

    return {
        "mapping": mapping, "feature_space": feat_name,
        "physics_metric": metric_name, "fold": fold_idx,
        "r2": r2, "n_train": len(Xtr), "n_test": len(Xte),
    }


def run_mapping(mapping: str, data: dict, regressor_kind: str,
                n_folds: int, max_train_rows: int | None,
                n_jobs: int, seed: int = 42) -> pd.DataFrame:
    print(f"\n── Mapping {mapping} ────────────────────────────────")
    labels   = data["labels"]
    features = data["features"]
    physics  = data["physics"]

    datasets = np.array(sorted(np.unique(labels)))
    n_ds = len(datasets)
    print(f"  {len(features)} feature spaces × {len(physics)} physics metrics")
    print(f"  {n_ds} datasets, {n_folds}-fold dataset-level CV")
    print(f"  pitch slot driven by: {PITCH_METRIC.get(mapping, '?')}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(datasets))

    args_list = []
    for feat_name, X in features.items():
        for metric_name, y in physics.items():
            for fold_idx, (tr_idx, te_idx) in enumerate(folds):
                args_list.append((
                    mapping, feat_name, metric_name,
                    X, y, labels,
                    datasets[tr_idx], datasets[te_idx], fold_idx,
                    regressor_kind, max_train_rows, seed,
                ))

    print(f"  Running {len(args_list)} fold-fits with {n_jobs} workers, "
          f"regressor={regressor_kind}...")
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_one_fold)(a) for a in args_list
    )
    df = pd.DataFrame(rows)

    print(f"\n  Mean R² (across {n_folds} folds), "
          f"feature space → physics metric:")
    summary = (df.dropna(subset=["r2"])
                  .groupby(["feature_space", "physics_metric"])["r2"]
                  .agg(["mean", "std"])
                  .reset_index())
    for fs in FEATURE_ORDER:
        if fs not in features:
            continue
        sub = summary[summary["feature_space"] == fs]
        line = f"    {FEATURE_LABELS.get(fs, fs):<14} "
        for m in PHYSICS_METRICS:
            row = sub[sub["physics_metric"] == m]
            if len(row) == 0:
                line += f" {m}=N/A"
            else:
                mean = row["mean"].values[0]
                std  = row["std"].values[0]
                line += f" {m}={mean:+.2f}±{std:.2f}"
        print(line)

    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_df: pd.DataFrame, out_dir: str):
    """One heatmap per physics metric. Rows = mappings, cols = feature
    spaces, cell value = mean R² across folds."""
    if all_df.empty:
        return

    metrics      = [m for m in PHYSICS_METRICS if m in all_df["physics_metric"].values]
    feat_spaces  = [f for f in FEATURE_ORDER if f in all_df["feature_space"].values]
    mappings     = sorted(all_df["mapping"].unique())

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(8, len(feat_spaces) * 1.6),
                                      max(2.5 * len(metrics), 6)),
                             dpi=DPI)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        n_maps  = len(mappings)
        n_feats = len(feat_spaces)
        mat = np.full((n_maps, n_feats), np.nan)
        std_mat = np.full((n_maps, n_feats), np.nan)
        for i, m in enumerate(mappings):
            for j, fs in enumerate(feat_spaces):
                sub = all_df[(all_df["mapping"] == m) &
                             (all_df["feature_space"] == fs) &
                             (all_df["physics_metric"] == metric)]["r2"].dropna()
                if len(sub) > 0:
                    mat[i, j]     = sub.mean()
                    std_mat[i, j] = sub.std()

        # cap the lower bound on the colour scale at 0 — negative R²
        # (worse than predicting the mean) is meaningful but noisy, and
        # mapping the colour bar there blows out the visible structure.
        clipped = np.clip(mat, 0.0, 1.0)
        im = ax.imshow(clipped, vmin=0.0, vmax=1.0,
                        cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_feats))
        ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in feat_spaces],
                           rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(n_maps))
        ax.set_yticklabels([f"Mapping {m}\n(pitch ← {PITCH_METRIC.get(m, '?')})"
                            for m in mappings], fontsize=9)
        for i in range(n_maps):
            for j in range(n_feats):
                v = mat[i, j]
                s = std_mat[i, j]
                if np.isfinite(v):
                    text = f"{v:+.2f}\n±{s:.2f}"
                    color = "white" if clipped[i, j] < 0.5 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=8, color=color)
        ax.set_title(f"Predicting {metric}  —  R² (clipped at 0)",
                     fontsize=11)
        plt.colorbar(im, ax=ax, label="Mean R² across folds")

    fig.suptitle(
        "Physics regression from audio features  (leave-datasets-out CV)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "physics_regression_r2.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: physics_regression_r2.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+", choices=MAPPINGS, default=MAPPINGS,
                    help="which mappings to evaluate (default: all)")
    ap.add_argument("--regressor", choices=["ridge", "rf"], default="ridge",
                    help="ridge (linear, fast) or rf (nonlinear, slower)")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="dataset-level KFold splits (default 5)")
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="cap on training rows per fold (default: no cap). "
                         "Useful for RF on very large mel/encodec features.")
    ap.add_argument("--n_jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                    help="parallel workers")
    args = ap.parse_args()

    out_dir = get_output_dir()
    print(f"Output dir : {out_dir}")
    print(f"Regressor  : {args.regressor}")
    print(f"Folds      : {args.n_folds}")
    print(f"Workers    : {args.n_jobs}")
    if args.max_train_rows:
        print(f"Train cap  : {args.max_train_rows} rows per fold")

    # ── load shared physics data once ──────────────────────────────────────
    selected_path = get_path("datasetSummaryPath")
    metrics_root  = get_path("metricspath")
    print(f"\nLoading physics from manifest...")
    all_physics = load_all_physics(metrics_root, selected_path)
    print(f"  {len(all_physics)} rows from "
          f"{all_physics['dataset'].nunique()} datasets")

    all_dfs = []
    for m in args.mapping:
        data = load_mapping_data(m, all_physics)
        if data is None:
            continue
        df = run_mapping(m, data, args.regressor, args.n_folds,
                         args.max_train_rows, args.n_jobs)
        all_dfs.append(df)

    if not all_dfs:
        print("No results produced.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "experiment_1b_results.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-fold R² → {csv_path}")

    # Compact summary table for the writeup
    print("\n── Mean R² (across folds) ───────────────────────────────────────")
    summary = (all_df.dropna(subset=["r2"])
                     .groupby(["mapping", "feature_space", "physics_metric"])["r2"]
                     .mean()
                     .reset_index()
                     .pivot_table(index=["mapping", "feature_space"],
                                   columns="physics_metric", values="r2"))
    print(summary.to_string())

    plot_results(all_df, out_dir)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()