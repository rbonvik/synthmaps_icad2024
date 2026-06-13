"""
experiment_1b_physics_regression.py
====================================
Experiment 1B: Recover physics metric values from audio features.

Question
--------
If a mapping faithfully encodes a physics metric in audio, then a
regressor trained on audio features should be able to predict that
metric's values on unseen datasets. Each of the six mappings A–F
routes one of {dipolar_energy, magnet_flips, hamming_from_init} to
each of the three FM parameters {pitch, harm_ratio, mod_index}, and
we ask which routing leads to the most recoverable audio.

Method
------
For each (mapping, feature_space, physics_metric):
  • Split the cleaned manifest into K folds at the dataset level.
  • For each fold:
      - Train a regressor on per-timestep (audio_features → metric)
        pairs from K-1 training folds.
      - Predict on the held-out fold (no timesteps from a test dataset
        ever seen in training).
      - Score with R² (coefficient of determination).
  • Report mean R² across folds.

R² interpretation:
  1.0   — perfect recovery
  0.0   — no better than predicting the mean
  < 0   — worse than the mean (model fails to generalise)

Dataset cleaning (applied before CV)
------------------------------------
  • Drop datasets with all-zero metric trajectories (broken sims).
  • Drop datasets whose triplet_std_sum is below --min_variance
    (default 0.2). Low-variance datasets make R² statistically unstable
    and contribute near-flat training rows that bias the regressor.
  • Group near-duplicate datasets so they always fall in the same CV
    fold. Two datasets are treated as duplicates when their
    (dipolar_energy_nstd, magnet_flips_nstd, hamming_from_init_nstd,
    energy_range) tuples differ by less than --dedup_eps in L1.

Why leave-DATASETS-out, not leave-timesteps-out?
  Random timestep splits leak. We want generalisation to *new* physical
  trajectories, so we hold out whole datasets — and group near-duplicates
  so the held-out fold is genuinely unseen.

Overall mapping ranking
-----------------------
After the per-metric heatmap, three "headline" scores are computed per
mapping. All three exclude `pitch` and `fm_params` (trivial baselines —
pitch IS one of the physics metrics by design; fm_params are the
synth's inputs and trivially recover the metrics that drive them).

  • mean_all          — mean R² across (perceptual, spectral, mel,
                        encodec, clap) × (3 physics metrics).
  • min_metric        — for each metric, average across feature spaces;
                        report the WORST-performing metric.
                        `min_metric_argmin` names the bottleneck.
  • handcrafted_mean  — same as mean_all but restricted to perceptual
                        and spectral. The "what a listener could
                        plausibly perceive" version.

By default these are computed on R² values clipped at 0 (negative R²
is "worse than the mean" — noise, not signal). Pass --no_clip to use
unclipped values, or --report_both to write a second ranking CSV with
the unclipped scores alongside.

Outputs (under <synthmapspath>/figures/evaluation/)
---------------------------------------------------
    physics_regression_r2.png        — heatmap of mean R² per
                                       (mapping, feature_space) for
                                       each physics metric.
    physics_regression_ranking.png   — three-panel bar chart of the
                                       headline scores.
    experiment_1b_results.csv        — per-(mapping, feature_space,
                                       physics_metric, fold) R².
    experiment_1b_ranking.csv        — per-mapping headline scores
                                       (clipped, default).
    experiment_1b_ranking_unclipped.csv
                                     — same with unclipped R² (only
                                       when --report_both is set).
    experiment_1b_cleaning.json      — log of which datasets were
                                       dropped or grouped, and why.

Usage
-----
    python experiment_1b_physics_regression.py
    python experiment_1b_physics_regression.py --regressor rf
    python experiment_1b_physics_regression.py --min_variance 0.3 --dedup_eps 0.002
    python experiment_1b_physics_regression.py --rank_only --report_both
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

# Per-metric column lists for the handcrafted feature CSVs.
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
FEATURE_ORDER = ["pitch", "fm_params", "perceptual", "spectral",
                 "mel", "encodec", "clap"]

# Excluded from headline rankings (trivial baselines).
RANKING_EXCLUDE = {"pitch", "fm_params"}
HANDCRAFTED_SPACES = {"perceptual", "spectral"}

# Which metric drives the pitch slot in each mapping. Used for figure
# labels only; the regression itself doesn't read this dict.
#   A=DHM, B=DMH → dipolar_energy → pitch
#   C=HDM, D=HMD → hamming_from_init → pitch
#   E=MDH, F=MHD → magnet_flips → pitch
PITCH_METRIC = {
    "A": "dipolar_energy",
    "B": "dipolar_energy",
    "C": "hamming_from_init",
    "D": "hamming_from_init",
    "E": "magnet_flips",
    "F": "magnet_flips",
}

# Feature loader registry: each entry says how to load one feature
# space from a mapping's results directory. Kept declarative so adding
# a new feature space is one entry rather than a new `if` block.
def _load_csv_cols(path: str, cols: list[str], n: int) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    avail = [c for c in cols if c in df.columns]
    return df[avail].reindex(range(n)).values

def _load_npy_flat(path: str, n: int) -> np.ndarray:
    arr = np.load(path)[:n]
    return arr.reshape(len(arr), -1) if arr.ndim > 2 else arr

FEATURE_LOADERS = {
    "perceptual": ("perceptual_csv",
                   lambda p, n: _load_csv_cols(p, PERCEPTUAL_COLS, n)),
    "spectral":   ("spectral_csv",
                   lambda p, n: _load_csv_cols(p, SPECTRAL_COLS, n)),
    "mel":        ("mel_npy",     lambda p, n: np.load(p)[:n]),
    "encodec":    ("encodec_npy", _load_npy_flat),
    "clap":       ("clap_npy",    lambda p, n: np.load(p)[:n]),
}

DPI = 200


# ── path helpers ──────────────────────────────────────────────────────────────

def get_mapping_paths(mapping: str) -> dict:
    root = get_path("synthmapspath")
    out  = os.path.join(root, "results", f"mapping_{mapping}")
    return {
        "params_csv":     os.path.join(root, "mapped_params",
                                       f"mapping_{mapping}",
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


# ── dataset cleaning ──────────────────────────────────────────────────────────
#
# The three cleaning steps are applied to the manifest BEFORE any data is
# loaded into the regression, so they keep entire datasets in or out of
# every (mapping, feature_space, metric) cell consistently.
# ─────────────────────────────────────────────────────────────────────────────

DEDUP_KEYS = ("dipolar_energy_nstd", "magnet_flips_nstd",
              "hamming_from_init_nstd", "energy_range")


def _is_zero_dataset(meta: dict) -> bool:
    """All three metric trajectories are flat (broken simulation)."""
    return (meta.get("triplet_std_sum", 0.0) == 0.0
            and meta.get("pct_zero_flip", 0.0) >= 99.9)


def _meta_key_vector(meta: dict) -> np.ndarray:
    return np.array([meta.get(k, 0.0) for k in DEDUP_KEYS], dtype=np.float64)


def _group_duplicates(metas: list[dict], eps: float) -> dict[str, int]:
    """Return {dataset_name: group_id}. Datasets within eps L1 distance
    on the four key columns share a group id."""
    if eps <= 0 or len(metas) < 2:
        return {m["name"]: i for i, m in enumerate(metas)}

    vecs = np.stack([_meta_key_vector(m) for m in metas])
    n = len(metas)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Quadratic scan — fine at n ~ 200. Switch to KDTree if it ever grows.
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(np.abs(vecs[i] - vecs[j])) < eps:
                union(i, j)

    root_to_gid: dict[int, int] = {}
    out: dict[str, int] = {}
    for i, m in enumerate(metas):
        r = find(i)
        if r not in root_to_gid:
            root_to_gid[r] = len(root_to_gid)
        out[m["name"]] = root_to_gid[r]
    return out


def clean_manifest(manifest_path: str, min_variance: float,
                   dedup_eps: float, log_path: str | None = None
                   ) -> tuple[list[dict], dict[str, int]]:
    """Apply cleaning to the manifest. Returns (kept_metas, group_ids)
    and optionally writes a JSON log of what was dropped/grouped."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    metas_all = manifest["datasets"]
    n_in = len(metas_all)

    dropped_zero = [m["name"] for m in metas_all if _is_zero_dataset(m)]
    after_zero = [m for m in metas_all if not _is_zero_dataset(m)]

    dropped_lowvar = [m["name"] for m in after_zero
                      if m.get("triplet_std_sum", 0.0) < min_variance]
    kept = [m for m in after_zero
            if m.get("triplet_std_sum", 0.0) >= min_variance]

    groups = _group_duplicates(kept, dedup_eps)
    n_groups = len(set(groups.values()))

    log = {
        "n_in_manifest":          n_in,
        "n_dropped_zero":         len(dropped_zero),
        "n_dropped_low_variance": len(dropped_lowvar),
        "n_kept":                 len(kept),
        "n_unique_groups":        n_groups,
        "min_variance_threshold": min_variance,
        "dedup_eps":              dedup_eps,
        "dropped_zero":           dropped_zero,
        "dropped_low_variance":   dropped_lowvar,
    }
    print(f"  Manifest cleaning:")
    print(f"    in:           {n_in} datasets")
    print(f"    dropped zero: {len(dropped_zero)}")
    print(f"    dropped <var: {len(dropped_lowvar)} (threshold "
          f"triplet_std_sum < {min_variance})")
    print(f"    kept:         {len(kept)} datasets in {n_groups} CV groups "
          f"(dedup_eps = {dedup_eps})")

    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"    log written to: {os.path.basename(log_path)}")

    return kept, groups


# ── data loading ──────────────────────────────────────────────────────────────

def _clean(X: np.ndarray) -> np.ndarray:
    """Replace inf/NaN with column medians (vectorised)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.where(np.isinf(X), np.nan, X)
    # nanmedian over rows; if a column is all-NaN, nanmedian warns and
    # returns NaN, which np.nan_to_num then turns into 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_medians = np.nanmedian(X, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.take(col_medians, np.where(mask)[1])
    return X


def _resolve_metrics_csv(metrics_csv: str, metrics_root: str,
                         name: str) -> str | None:
    for c in [metrics_csv,
              os.path.join(metrics_root, metrics_csv),
              os.path.join(metrics_root, os.path.basename(metrics_csv)),
              os.path.join(metrics_root, f"{name}.csv")]:
        if os.path.exists(c):
            return c
    return None


def load_all_physics(metrics_root: str, metas: list[dict]) -> pd.DataFrame:
    """Load per-timestep physics metric CSVs for the cleaned manifest."""
    frames = []
    n_missing = 0
    for meta in metas:
        path = _resolve_metrics_csv(meta["metrics_csv"], metrics_root,
                                    meta["name"])
        if path is None:
            print(f"  [warn] missing physics CSV for {meta['name']}")
            n_missing += 1
            continue
        df = pd.read_csv(path)
        df["dataset"] = meta["name"]
        keep = ["dataset", "time"] + [m for m in PHYSICS_METRICS
                                      if m in df.columns]
        frames.append(df[keep])
    if not frames:
        raise RuntimeError("No physics CSVs found.")
    if n_missing:
        print(f"  [warn] {n_missing} CSVs missing")
    return pd.concat(frames, ignore_index=True)


def load_mapping_data(mapping: str, all_physics: pd.DataFrame,
                      kept_names: set[str]) -> dict | None:
    """Load FM params + audio features for one mapping, restricted to
    the cleaned manifest. Returns None if the params CSV is missing."""
    paths = get_mapping_paths(mapping)
    if not os.path.exists(paths["params_csv"]):
        print(f"  [skip] params CSV missing for {mapping}")
        return None

    params_df = pd.read_csv(paths["params_csv"]).reset_index(drop=True)
    if "dataset" not in params_df.columns or "time" not in params_df.columns:
        raise ValueError(f"params CSV for {mapping} needs 'dataset' and 'time'")
    keep_mask = params_df["dataset"].isin(kept_names).values
    N_full = len(params_df)

    phys_join = params_df[["dataset", "time"]].merge(
        all_physics, on=["dataset", "time"], how="left",
    )
    physics = {m: phys_join[m].values.astype(np.float64)[keep_mask]
               for m in PHYSICS_METRICS if m in phys_join.columns}

    features: dict[str, np.ndarray] = {}

    # Baselines: pitch alone, and the full FM parameter triple.
    midi = frequency2midi(params_df["freq"].values.astype(np.float64))
    features["pitch"] = _clean(midi.reshape(-1, 1))[keep_mask]
    fm = np.column_stack([midi,
                          params_df["harm_ratio"].values,
                          params_df["mod_index"].values])
    features["fm_params"] = _clean(fm)[keep_mask]

    # Audio feature spaces, loaded via the registry.
    for name, (path_key, loader) in FEATURE_LOADERS.items():
        path = paths[path_key]
        if not os.path.exists(path):
            continue
        try:
            arr = loader(path, N_full)
        except Exception as e:
            print(f"  [warn] failed to load {name} for {mapping}: {e}")
            continue
        features[name] = _clean(arr)[keep_mask]

    return {
        "labels":   params_df["dataset"].values[keep_mask],
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
        return {"mapping": mapping, "feature_space": feat_name,
                "physics_metric": metric_name, "fold": fold_idx,
                "r2": np.nan, "n_train": len(Xtr), "n_test": len(Xte)}

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

    return {"mapping": mapping, "feature_space": feat_name,
            "physics_metric": metric_name, "fold": fold_idx,
            "r2": r2, "n_train": len(Xtr), "n_test": len(Xte)}


def build_dataset_folds(datasets: np.ndarray, groups: dict[str, int],
                        n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """KFold over duplicate-groups, then translate group indices back to
    dataset names. Guarantees near-duplicate datasets land in the same
    fold (either both train or both test)."""
    gid_of = np.array([groups[d] for d in datasets])
    unique_gids = np.array(sorted(set(gid_of)))
    kf = KFold(n_splits=min(n_folds, len(unique_gids)),
               shuffle=True, random_state=seed)
    folds = []
    for tr_gid_idx, te_gid_idx in kf.split(unique_gids):
        tr_gids = set(unique_gids[tr_gid_idx])
        te_gids = set(unique_gids[te_gid_idx])
        tr_datasets = datasets[np.isin(gid_of, list(tr_gids))]
        te_datasets = datasets[np.isin(gid_of, list(te_gids))]
        folds.append((tr_datasets, te_datasets))
    return folds


def run_mapping(mapping: str, data: dict, groups: dict[str, int],
                regressor_kind: str, n_folds: int,
                max_train_rows: int | None, n_jobs: int,
                seed: int = 42) -> pd.DataFrame:
    print(f"\n── Mapping {mapping} ────────────────────────────────")
    labels   = data["labels"]
    features = data["features"]
    physics  = data["physics"]

    datasets = np.array(sorted(np.unique(labels)))
    n_groups = len(set(groups[d] for d in datasets))
    print(f"  {len(features)} feature spaces × {len(physics)} physics metrics")
    print(f"  {len(datasets)} datasets in {n_groups} CV groups, "
          f"{n_folds}-fold group-level CV")
    print(f"  pitch slot driven by: {PITCH_METRIC.get(mapping, '?')}")

    folds = build_dataset_folds(datasets, groups, n_folds, seed)

    all_rows: list[dict] = []
    for feat_name in FEATURE_ORDER:
        if feat_name not in features:
            continue
        X = features[feat_name]
        print(f"  ── feature space: {FEATURE_LABELS.get(feat_name, feat_name)} "
              f"(shape {X.shape}, {X.nbytes / 1e9:.2f} GB)")

        args_list = []
        for metric_name, y in physics.items():
            for fold_idx, (tr_ds, te_ds) in enumerate(folds):
                args_list.append((
                    mapping, feat_name, metric_name,
                    X, y, labels, tr_ds, te_ds, fold_idx,
                    regressor_kind, max_train_rows, seed,
                ))
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(run_one_fold)(a) for a in args_list
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

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
                line += f" {m}={row['mean'].values[0]:+.2f}±{row['std'].values[0]:.2f}"
        print(line)

    return df


# ── overall mapping ranking ───────────────────────────────────────────────────

def compute_ranking(all_df: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
    """
    Per-mapping headline scores: mean_all, min_metric, handcrafted_mean.
    Negative R² ("worse than the mean") is noise; by default we clip at 0
    so a few bad folds don't drag the mean into negative territory in a
    way that's hard to interpret. Pass clip=False to use raw R² values.
    """
    if all_df.empty:
        return pd.DataFrame()

    mean_cells = (all_df.dropna(subset=["r2"])
                        .groupby(["mapping", "feature_space",
                                  "physics_metric"])
                        ["r2"].mean()
                        .reset_index())
    if clip:
        mean_cells["r2"] = mean_cells["r2"].clip(lower=0.0)

    rows = []
    for mapping in sorted(mean_cells["mapping"].unique()):
        sub = mean_cells[mean_cells["mapping"] == mapping]
        non_trivial = sub[~sub["feature_space"].isin(RANKING_EXCLUDE)]
        handcrafted = sub[sub["feature_space"].isin(HANDCRAFTED_SPACES)]

        mean_all = non_trivial["r2"].mean() if len(non_trivial) else np.nan

        if len(non_trivial):
            per_metric = non_trivial.groupby("physics_metric")["r2"].mean()
            min_metric_val    = float(per_metric.min())
            min_metric_argmin = str(per_metric.idxmin())
        else:
            min_metric_val, min_metric_argmin = np.nan, None

        handcrafted_mean = handcrafted["r2"].mean() if len(handcrafted) else np.nan

        rows.append({
            "mapping":           mapping,
            "mean_all":          mean_all,
            "min_metric":        min_metric_val,
            "min_metric_argmin": min_metric_argmin,
            "handcrafted_mean":  handcrafted_mean,
        })

    df = pd.DataFrame(rows)
    df["composite"] = df[["mean_all", "min_metric",
                          "handcrafted_mean"]].mean(axis=1)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    return df


def print_ranking(rank_df: pd.DataFrame, label: str = "clipped"):
    if rank_df.empty:
        return
    print("\n" + "=" * 78)
    print(f"  OVERALL MAPPING RANKING  ({label})")
    print("  (pitch and fm_params excluded — trivial baselines)")
    print("=" * 78)
    print(f"  {'rank':<5} {'mapping':<10} {'mean_all':>10} "
          f"{'min_metric':>12} {'(bottleneck)':<22} {'handcrafted':>12}")
    print("  " + "-" * 76)
    for i, row in rank_df.iterrows():
        bn = row["min_metric_argmin"] or "—"
        print(f"  {i+1:<5} {row['mapping']:<10} "
              f"{row['mean_all']:>10.3f} "
              f"{row['min_metric']:>12.3f} "
              f"{'('+bn+')':<22} "
              f"{row['handcrafted_mean']:>12.3f}")
    print("=" * 78)
    winner = rank_df.iloc[0]
    print(f"\n  Best overall ({label}): Mapping {winner['mapping']}  "
          f"(composite = {winner['composite']:.3f})")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_df: pd.DataFrame, out_dir: str, clip: bool = True):
    """One heatmap per physics metric."""
    if all_df.empty:
        return
    metrics     = [m for m in PHYSICS_METRICS if m in all_df["physics_metric"].values]
    feat_spaces = [f for f in FEATURE_ORDER if f in all_df["feature_space"].values]
    mappings    = sorted(all_df["mapping"].unique())

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(8, len(feat_spaces) * 1.6),
                                      max(2.5 * len(metrics), 6)),
                             dpi=DPI)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        n_maps, n_feats = len(mappings), len(feat_spaces)
        mat = np.full((n_maps, n_feats), np.nan)
        std_mat = np.full((n_maps, n_feats), np.nan)
        for i, m in enumerate(mappings):
            for j, fs in enumerate(feat_spaces):
                sub = all_df[(all_df["mapping"] == m) &
                             (all_df["feature_space"] == fs) &
                             (all_df["physics_metric"] == metric)
                            ]["r2"].dropna()
                if len(sub):
                    mat[i, j], std_mat[i, j] = sub.mean(), sub.std()

        clipped = np.clip(mat, 0.0, 1.0) if clip else mat
        vmin = 0.0 if clip else float(np.nanmin(clipped))
        im = ax.imshow(clipped, vmin=vmin, vmax=1.0,
                       cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_feats))
        ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in feat_spaces],
                           rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(n_maps))
        ax.set_yticklabels(
            [f"Mapping {m}\n(pitch ← {PITCH_METRIC.get(m, '?')})"
             for m in mappings], fontsize=9)
        for i in range(n_maps):
            for j in range(n_feats):
                v, s = mat[i, j], std_mat[i, j]
                if np.isfinite(v):
                    color = "white" if clipped[i, j] < 0.5 * (1.0 + vmin) else "black"
                    ax.text(j, i, f"{v:+.2f}\n±{s:.2f}",
                            ha="center", va="center", fontsize=8, color=color)
        title_suffix = "R² (clipped at 0)" if clip else "R² (unclipped)"
        ax.set_title(f"Predicting {metric}  —  {title_suffix}", fontsize=11)
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


def plot_ranking(rank_df: pd.DataFrame, out_dir: str):
    if rank_df.empty:
        return
    panels = [
        ("mean_all",         "Mean R² across all metrics & feature spaces",
         "Higher = more balanced overall recovery"),
        ("min_metric",       "Worst-metric R²  (bottleneck)",
         "Higher = no single metric is dropped"),
        ("handcrafted_mean", "Mean R²  —  handcrafted features only",
         "What a listener could plausibly perceive"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)
    for ax, (col, title, subtitle) in zip(axes, panels):
        sub = rank_df.sort_values(col, ascending=False).reset_index(drop=True)
        x = np.arange(len(sub))
        vals = sub[col].values
        bars = ax.bar(x, vals, color="#2b6cb0",
                      edgecolor="black", linewidth=0.5, alpha=0.9)
        if len(bars):
            bars[0].set_color("#c2410c")
            bars[0].set_alpha(0.95)
        for j, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(x[j], v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Mapping {m}" for m in sub["mapping"]],
                           rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, min(1.0,
                           float(np.nanmax(vals)) * 1.18 if len(vals) else 1.0))
        ax.set_ylabel("Mean R² (clipped at 0)")
        ax.set_title(title, fontsize=11)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, style="italic", color="#555")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if col == "min_metric":
            for j, xpos in enumerate(x):
                bn = sub.iloc[j]["min_metric_argmin"]
                if isinstance(bn, str):
                    ax.text(xpos, vals[j] / 2, bn,
                            ha="center", va="center", fontsize=7,
                            color="white", rotation=90)
    fig.suptitle("Overall mapping ranking — three views\n"
                 "(pitch and fm_params excluded as trivial baselines)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "physics_regression_ranking.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: physics_regression_ranking.png")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+", choices=MAPPINGS, default=MAPPINGS,
                    help="which mappings to evaluate (default: all)")
    ap.add_argument("--regressor", choices=["ridge", "rf"], default="ridge",
                    help="ridge (linear, fast) or rf (nonlinear, slower)")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="dataset-level KFold splits (default 5)")
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="cap on training rows per fold (default: no cap)")
    ap.add_argument("--skip_features", nargs="+",
                    choices=FEATURE_ORDER, default=[],
                    help="feature spaces to skip")
    ap.add_argument("--n_jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                    help="parallel workers")
    ap.add_argument("--min_variance", type=float, default=0.2,
                    help="drop datasets with triplet_std_sum below this "
                         "(default 0.2). Set to 0 to disable.")
    ap.add_argument("--dedup_eps", type=float, default=0.001,
                    help="L1 threshold on (dipolar/flips/hamming nstd + "
                         "energy_range) for grouping near-duplicate datasets "
                         "into the same CV fold (default 0.001). Set to 0 "
                         "to disable grouping.")
    ap.add_argument("--no_clip", action="store_true",
                    help="report unclipped R² in the ranking (allows negative).")
    ap.add_argument("--report_both", action="store_true",
                    help="write a second ranking CSV with the alternative "
                         "(clipped vs unclipped) scores alongside.")
    ap.add_argument("--rank_only", action="store_true",
                    help="skip regression; read existing experiment_1b_results.csv "
                         "and just recompute the ranking.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = get_output_dir()
    print(f"Output dir     : {out_dir}")
    print(f"Regressor      : {args.regressor}")
    print(f"Folds          : {args.n_folds}")
    print(f"Workers        : {args.n_jobs}")
    print(f"Clip ranking R²: {not args.no_clip}")
    if args.max_train_rows:
        print(f"Train cap      : {args.max_train_rows} rows per fold")
    if args.skip_features:
        print(f"Skipped feats  : {args.skip_features}")

    csv_path  = os.path.join(out_dir, "experiment_1b_results.csv")
    rank_path = os.path.join(out_dir, "experiment_1b_ranking.csv")
    log_path  = os.path.join(out_dir, "experiment_1b_cleaning.json")

    if args.rank_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"--rank_only requires existing {csv_path}; run without it first.")
        print(f"\nLoading existing results from {csv_path}")
        all_df = pd.read_csv(csv_path)
    else:
        selected_path = get_path("datasetSummaryPath")
        metrics_root  = get_path("metricspath")

        print("\nCleaning manifest...")
        kept_metas, groups = clean_manifest(
            selected_path, args.min_variance, args.dedup_eps, log_path,
        )
        kept_names = {m["name"] for m in kept_metas}

        print(f"\nLoading physics from {len(kept_metas)} kept datasets...")
        all_physics = load_all_physics(metrics_root, kept_metas)
        print(f"  {len(all_physics)} rows from "
              f"{all_physics['dataset'].nunique()} datasets")

        all_dfs = []
        for m in args.mapping:
            data = load_mapping_data(m, all_physics, kept_names)
            if data is None:
                continue
            for fs in args.skip_features:
                data["features"].pop(fs, None)
            df = run_mapping(m, data, groups,
                             args.regressor, args.n_folds,
                             args.max_train_rows, args.n_jobs)
            all_dfs.append(df)

        if not all_dfs:
            print("No results produced.")
            return

        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df.to_csv(csv_path, index=False)
        print(f"\nSaved per-fold R² → {csv_path}")

    # ── per-metric summary table ──────────────────────────────────────────
    print("\n── Mean R² (across folds) ──────────────────────────────────────")
    summary = (all_df.dropna(subset=["r2"])
                     .groupby(["mapping", "feature_space",
                               "physics_metric"])["r2"]
                     .mean().reset_index()
                     .pivot_table(index=["mapping", "feature_space"],
                                   columns="physics_metric", values="r2"))
    print(summary.to_string())

    # ── ranking (default clipped, optional unclipped twin) ────────────────
    clip_primary = not args.no_clip
    primary_label = "clipped" if clip_primary else "unclipped"

    rank_df = compute_ranking(all_df, clip=clip_primary)
    print_ranking(rank_df, label=primary_label)
    rank_df.to_csv(rank_path, index=False)
    print(f"\nSaved ranking ({primary_label}) → {rank_path}")

    if args.report_both:
        alt_clip = not clip_primary
        alt_label = "clipped" if alt_clip else "unclipped"
        alt_rank = compute_ranking(all_df, clip=alt_clip)
        print_ranking(alt_rank, label=alt_label)
        alt_path = os.path.join(out_dir,
                                f"experiment_1b_ranking_{alt_label}.csv")
        alt_rank.to_csv(alt_path, index=False)
        print(f"\nSaved ranking ({alt_label}) → {alt_path}")

    # ── figures ───────────────────────────────────────────────────────────
    plot_results(all_df, out_dir, clip=clip_primary)
    plot_ranking(rank_df, out_dir)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()