"""
evaluate_sonification.py
========================
Quantitative evaluation of the ASI → FM-synth sonification mappings.

Two evaluation axes
-------------------
1.  Structure preservation (within-dataset correlation)
    For each dataset, compute Spearman rank correlation between every
    physics metric and every individual audio feature. Then average
    across datasets.

    Handcrafted features (perceptual + spectral) get one ρ per
    (physics metric × named audio feature) cell — a fully interpretable
    heatmap. The thing on the axis is named: "brightness",
    "spectral centroid", etc.

    High-dimensional feature spaces (Mel, EnCodec, CLAP) have learned,
    unnamed axes, so we summarise each one by max |ρ| across its axes per
    physics metric. This says "the best linear-rank relationship this
    feature space affords for this physics metric — locally".

    Three 1-D parameter baselines are included as within-mapping controls:
    correlation between each raw FM synth parameter (pitch, harm_ratio,
    mod_index) and each physics metric. These are diagnostics of routing
    fidelity: how well does the physics metric survive *before* the synth
    turns it into sound? The gap between a parameter baseline and the
    best audio feature for the same physics metric quantifies the
    non-proportionality introduced by FM synthesis itself. Since each
    mapping routes a different physics metric to each parameter, these
    baselines are not comparable across mappings.

2.  Auditory event alignment
    For each dataset, correlate |Δphysics_metric| with |Δaudio_feature|
    across time. Reported alongside a time-shuffled baseline so the
    reader can see how much of the alignment is structure vs. coincidence
    of marginals.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_path, frequency2midi

# ── config ────────────────────────────────────────────────────────────────────

PHYSICS_COLS = ["dipolar_energy", "magnet_flips", "hamming_from_init"]

PERCEPTUAL_COLS = [
    "hardness", "depth", "brightness", "roughness", "warmth", "sharpness", "boominess"
]
SPECTRAL_COLS = [
    "spectral_centroid", "spectral_crest", "spectral_decrease", "spectral_energy",
    "spectral_flatness", "spectral_kurtosis", "spectral_roll_off", "spectral_skewness",
    "spectral_slope", "spectral_spread", "inharmonicity",
]

HANDCRAFTED_SPACES = ["perceptual", "spectral"]
LEARNED_SPACES = ["mel", "encodec", "clap"]

# Direct FM parameters used as routing-fidelity baselines.
PARAM_BASELINE_SPACES = ["pitch", "harm_ratio", "mod_index"]

ALIGNMENT_PAIRS = [
    ("dipolar_energy",    "brightness"),
    ("dipolar_energy",    "spectral_centroid"),
    ("magnet_flips",      "roughness"),
    ("magnet_flips",      "sharpness"),
    ("hamming_from_init", "warmth"),
    ("hamming_from_init", "spectral_spread"),
]

FEATURE_LABELS = {
    "pitch":      "Pitch only",
    "fm_params":  "FM params",
    "perceptual": "Perceptual",
    "spectral":   "Spectral",
    "mel":        "Mel spec.",
    "encodec":    "EnCodec",
    "clap":       "CLAP",
    "harm_ratio": "Harm. ratio",
    "mod_index":  "Mod. index",
}

PHYSICS_LABELS = {
    "dipolar_energy":    "Dipolar energy",
    "magnet_flips":      "Magnet flips",
    "hamming_from_init": "Hamming dist. from init",
}

MAPPINGS = ["A", "B", "C", "D", "E", "F"]
DPI = 200

MAPPING_ORDER  = ["A", "B", "C", "D", "E", "F"]
MAPPING_COLORS = {
    "A":        "#2b6cb0",
    "B":        "#c2410c",
    "C":        "#059669",
    "D":        "#dc2626",
    "E":        "#7c3aed",
    "F":        "#ea580c",
}


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

def load_manifest():
    selected_path = get_path("datasetSummaryPath")
    metrics_root  = get_path("metricspath")
    with open(selected_path) as f:
        manifest = json.load(f)
    return manifest["datasets"], metrics_root


def _resolve_metrics_csv(metrics_csv: str, metrics_root: str, name: str) -> str | None:
    candidates = [
        metrics_csv,
        os.path.join(metrics_root, metrics_csv),
        os.path.join(metrics_root, os.path.basename(metrics_csv)),
        os.path.join(metrics_root, f"{name}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_all_physics(datasets: list, metrics_root: str) -> pd.DataFrame:
    frames = []
    n_missing = 0
    for meta in datasets:
        path = _resolve_metrics_csv(meta["metrics_csv"], metrics_root, meta["name"])
        if path is None:
            print(f"  [warn] missing physics CSV for {meta['name']} "
                  f"(tried under {metrics_root})")
            n_missing += 1
            continue
        df = pd.read_csv(path)
        df["dataset"] = meta["name"]
        keep = ["dataset", "time"] + [c for c in PHYSICS_COLS if c in df.columns]
        frames.append(df[keep])
    if not frames:
        raise RuntimeError(
            f"No physics CSVs found. Checked {len(datasets)} manifest "
            f"entries under '{metrics_root}'."
        )
    if n_missing:
        print(f"  [warn] {n_missing} of {len(datasets)} physics CSVs "
              f"could not be located")
    return pd.concat(frames, ignore_index=True)


def _normalise(X: np.ndarray) -> np.ndarray:
    """Clip 5–95th-percentile outliers, fill NaN with column median, MinMax scale."""
    X = X.astype(float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
        col = np.clip(col, lo, hi)
        med = float(np.nanmedian(col))
        col = np.where(np.isnan(col), med, col)
        X[:, j] = col
    return MinMaxScaler().fit_transform(X)


def load_mapping_data(paths: dict, all_physics: pd.DataFrame) -> dict | None:
    """
    Returns:
      params_df  — params CSV.
      phys_X     — (N, P) physics array, NaN where join failed.
      phys_cols  — list of physics column names.
      named      — dict feature_space → DataFrame (named audio columns
                   + dataset + time). Used for handcrafted heatmap +
                   parameter baselines (pitch, harm_ratio, mod_index).
      learned    — dict feature_space → (N, D) normalised array. Used for
                   "best |ρ| across axes" summary.
    """
    if not os.path.exists(paths["params_csv"]):
        print(f"  [skip] params CSV not found: {paths['params_csv']}")
        return None

    params_df = pd.read_csv(paths["params_csv"]).reset_index(drop=True)
    N = len(params_df)

    phys_cols = [c for c in PHYSICS_COLS if c in all_physics.columns]
    if "dataset" in params_df.columns and "time" in params_df.columns:
        merged = params_df[["dataset", "time"]].merge(
            all_physics[["dataset", "time"] + phys_cols],
            on=["dataset", "time"], how="left",
        )
        phys_X = merged[phys_cols].values.astype(float)
    else:
        phys_X = np.full((N, len(phys_cols)), np.nan)

    named   = {}
    learned = {}

    # Parameter baselines as single-column named "spaces".
    # pitch goes through MIDI conversion (carrier freq is exponential, perceived linear);
    # harm_ratio and mod_index are already on a linear scale (see 01_build_synth_params.py).
    midi = frequency2midi(params_df["freq"].values.astype(np.float64))
    pitch_df = pd.DataFrame({"pitch": midi})
    if "dataset" in params_df.columns:
        pitch_df["dataset"] = params_df["dataset"].values
    if "time" in params_df.columns:
        pitch_df["time"] = params_df["time"].values
    named["pitch"] = pitch_df

    for param_col in ("harm_ratio", "mod_index"):
        if param_col not in params_df.columns:
            continue
        baseline_df = pd.DataFrame({param_col: params_df[param_col].values.astype(float)})
        if "dataset" in params_df.columns:
            baseline_df["dataset"] = params_df["dataset"].values
        if "time" in params_df.columns:
            baseline_df["time"] = params_df["time"].values
        named[param_col] = baseline_df

    if os.path.exists(paths["perceptual_csv"]):
        raw = pd.read_csv(paths["perceptual_csv"], index_col=0)
        avail = [c for c in PERCEPTUAL_COLS if c in raw.columns]
        aligned = raw[avail].reindex(range(N)).copy()
        if "dataset" in params_df.columns:
            aligned["dataset"] = params_df["dataset"].values
        if "time" in params_df.columns:
            aligned["time"] = params_df["time"].values
        named["perceptual"] = aligned

    if os.path.exists(paths["spectral_csv"]):
        raw = pd.read_csv(paths["spectral_csv"], index_col=0)
        avail = [c for c in SPECTRAL_COLS if c in raw.columns]
        aligned = raw[avail].reindex(range(N)).copy()
        if "dataset" in params_df.columns:
            aligned["dataset"] = params_df["dataset"].values
        if "time" in params_df.columns:
            aligned["time"] = params_df["time"].values
        named["spectral"] = aligned

    if os.path.exists(paths["mel_npy"]):
        mels = np.load(paths["mel_npy"])[:N]
        learned["mel"] = _normalise(mels)

    if os.path.exists(paths["encodec_npy"]):
        embs = np.load(paths["encodec_npy"])[:N]
        learned["encodec"] = _normalise(embs.reshape(len(embs), -1))

    if os.path.exists(paths["clap_npy"]):
        embs = np.load(paths["clap_npy"])[:N]
        learned["clap"] = _normalise(embs)

    return {
        "params_df": params_df,
        "phys_X":    phys_X,
        "phys_cols": phys_cols,
        "named":     named,
        "learned":   learned,
    }


# ── correlation utilities ─────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = spearmanr(a, b)
    rho = r.statistic if hasattr(r, "statistic") else r.correlation
    return float(rho)


def per_dataset_correlation(params_df: pd.DataFrame,
                            phys_X: np.ndarray,
                            phys_cols: list,
                            audio_X: np.ndarray) -> np.ndarray:
    """
    For one named audio feature (1-D), compute Spearman ρ vs each physics
    metric per dataset, then average across datasets.

    Returns (P, 3): mean_rho, std_rho, n_datasets for each physics metric.
    """
    P = len(phys_cols)
    out = np.full((P, 3), np.nan)
    out[:, 2] = 0

    if "dataset" not in params_df.columns:
        return out

    per_metric = {j: [] for j in range(P)}
    for ds in params_df["dataset"].unique():
        mask = (params_df["dataset"] == ds).values
        if mask.sum() < 4:
            continue
        a = audio_X[mask]
        for j in range(P):
            p = phys_X[mask, j]
            valid = np.isfinite(p) & np.isfinite(a)
            if valid.sum() < 4:
                continue
            rho = _spearman(p[valid], a[valid])
            if np.isfinite(rho):
                per_metric[j].append(rho)

    for j in range(P):
        rhos = per_metric[j]
        if rhos:
            out[j, 0] = float(np.mean(rhos))
            out[j, 1] = float(np.std(rhos))
            out[j, 2] = len(rhos)
    return out


# ── Section 1a: handcrafted correlation table ─────────────────────────────────

def run_handcrafted_correlation(data: dict, mapping: str) -> pd.DataFrame:
    rows = []
    params_df = data["params_df"]
    phys_X    = data["phys_X"]
    phys_cols = data["phys_cols"]

    for space in HANDCRAFTED_SPACES + PARAM_BASELINE_SPACES:
        if space not in data["named"]:
            continue
        df = data["named"][space]
        feat_cols = [c for c in df.columns if c not in ("dataset", "time")]
        for fc in feat_cols:
            audio_X = df[fc].values.astype(float)
            stats = per_dataset_correlation(params_df, phys_X, phys_cols, audio_X)
            for j, pc in enumerate(phys_cols):
                rows.append({
                    "mapping":        mapping,
                    "feature_space":  space,
                    "audio_feature":  fc,
                    "physics_metric": pc,
                    "mean_rho":       stats[j, 0],
                    "std_rho":        stats[j, 1],
                    "n_datasets":     int(stats[j, 2]),
                })
    return pd.DataFrame(rows)


# ── Section 1b: learned-space best |ρ| across axes ────────────────────────────

def run_learned_correlation(data: dict, mapping: str) -> pd.DataFrame:
    """
    Per dataset, per learned feature space, per physics metric:
      take the max |ρ| across all axes.
    Then average those across datasets.
    """
    rows = []
    params_df = data["params_df"]
    phys_X    = data["phys_X"]
    phys_cols = data["phys_cols"]

    if "dataset" not in params_df.columns:
        return pd.DataFrame()

    for space in LEARNED_SPACES:
        if space not in data["learned"]:
            continue
        X = data["learned"][space]
        per_metric = {pc: [] for pc in phys_cols}

        for ds in params_df["dataset"].unique():
            mask = (params_df["dataset"] == ds).values
            if mask.sum() < 4:
                continue
            X_sub = X[mask]
            for j, pc in enumerate(phys_cols):
                p = phys_X[mask, j]
                valid = np.isfinite(p)
                if valid.sum() < 4:
                    continue
                p_v = p[valid]
                best = 0.0
                for d in range(X_sub.shape[1]):
                    a = X_sub[valid, d]
                    if np.std(a) == 0:
                        continue
                    rho = _spearman(p_v, a)
                    if np.isfinite(rho) and abs(rho) > best:
                        best = abs(rho)
                if best > 0:
                    per_metric[pc].append(best)

        for pc in phys_cols:
            rhos = per_metric[pc]
            rows.append({
                "mapping":        mapping,
                "feature_space":  space,
                "physics_metric": pc,
                "best_abs_rho":   float(np.mean(rhos)) if rhos else np.nan,
                "std":            float(np.std(rhos))  if rhos else np.nan,
                "n_datasets":     len(rhos),
            })
    return pd.DataFrame(rows)


# ── Section 2: Auditory Event Alignment ───────────────────────────────────────

def _align_rho(pd_arr: np.ndarray, ad_arr: np.ndarray) -> tuple[float, int]:
    n = min(len(pd_arr), len(ad_arr))
    if n < 3:
        return np.nan, 0
    valid = np.isfinite(pd_arr[:n]) & np.isfinite(ad_arr[:n])
    if valid.sum() < 3:
        return np.nan, int(valid.sum())
    rho = _spearman(pd_arr[:n][valid], ad_arr[:n][valid])
    return rho, int(valid.sum())


def _shuffled_align_rho(pd_arr: np.ndarray, ad_arr: np.ndarray,
                        n_shuffles: int, rng: np.random.Generator) -> float:
    n = min(len(pd_arr), len(ad_arr))
    if n < 3:
        return np.nan
    valid = np.isfinite(pd_arr[:n]) & np.isfinite(ad_arr[:n])
    if valid.sum() < 3:
        return np.nan
    p = pd_arr[:n][valid]
    a = ad_arr[:n][valid]
    rhos = []
    for _ in range(n_shuffles):
        a_shuf = rng.permutation(a)
        r = _spearman(p, a_shuf)
        if np.isfinite(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan


def run_event_alignment(data: dict, all_physics: pd.DataFrame,
                        mapping: str, n_shuffles: int) -> pd.DataFrame:
    params_df = data["params_df"]
    if "dataset" not in params_df.columns or "time" not in params_df.columns:
        print("  [skip] params CSV has no 'dataset'/'time' columns")
        return pd.DataFrame()

    phys_cols = data["phys_cols"]
    merged = params_df.reset_index(drop=True).merge(
        all_physics[["dataset", "time"] + phys_cols],
        on=["dataset", "time"], how="left",
    )
    merged.index = range(len(merged))

    rng = np.random.default_rng(42)
    rows = []
    for ds in params_df["dataset"].unique():
        ds_mask  = (merged["dataset"] == ds).values
        orig_idx = np.where(ds_mask)[0]
        sub      = merged.iloc[orig_idx].sort_values("time").reset_index()

        if len(sub) < 4:
            continue

        phys_diffs = {}
        for pc in phys_cols:
            v = sub[pc].values.astype(float)
            if np.all(~np.isfinite(v)):
                continue
            phys_diffs[pc] = np.abs(np.diff(v))

        audio_diffs = {}
        for feat_name in HANDCRAFTED_SPACES:
            if feat_name not in data["named"]:
                continue
            feat_df = data["named"][feat_name]
            feat_cols = [c for c in feat_df.columns if c not in ("dataset", "time")]
            feat_sub  = feat_df.iloc[sub["index"].values]
            for fc in feat_cols:
                v = feat_sub[fc].values.astype(float)
                audio_diffs[(feat_name, fc)] = np.abs(np.diff(v))

        for pc, pd_arr in phys_diffs.items():
            for (feat_name, fc), ad_arr in audio_diffs.items():
                if (pc, fc) not in ALIGNMENT_PAIRS:
                    continue
                rho_real, n = _align_rho(pd_arr, ad_arr)
                if not np.isfinite(rho_real):
                    continue
                rho_shuf = _shuffled_align_rho(pd_arr, ad_arr, n_shuffles, rng)
                rows.append({
                    "mapping":         mapping,
                    "dataset":         ds,
                    "physics_metric":  pc,
                    "audio_feature":   fc,
                    "feature_space":   feat_name,
                    "rho":             rho_real,
                    "rho_shuffled":    rho_shuf,
                    "rho_excess":      rho_real - rho_shuf if np.isfinite(rho_shuf) else np.nan,
                    "n":               n,
                })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_handcrafted_heatmaps(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return

    sub = df[df["feature_space"].isin(HANDCRAFTED_SPACES)]
    if sub.empty:
        return

    mappings = sorted(sub["mapping"].unique(),
                      key=lambda m: MAPPING_ORDER.index(m) if m in MAPPING_ORDER else 99)

    cols = []
    for space in HANDCRAFTED_SPACES:
        cols.extend(sorted(sub.loc[sub["feature_space"] == space,
                                   "audio_feature"].unique()))
    rows = [pc for pc in PHYSICS_COLS if pc in sub["physics_metric"].values]

    for m in mappings:
        sm = sub[sub["mapping"] == m]
        mat = np.full((len(rows), len(cols)), np.nan)
        for i, pc in enumerate(rows):
            for j, fc in enumerate(cols):
                cell = sm[(sm["physics_metric"] == pc) &
                          (sm["audio_feature"]  == fc)]
                if not cell.empty:
                    mat[i, j] = cell["mean_rho"].values[0]

        fig_w = max(8, 0.55 * len(cols) + 3)
        fig_h = max(3, 0.7 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

        n_perc = sum(1 for fc in cols
                     if fc in sub.loc[sub["feature_space"] == "perceptual",
                                      "audio_feature"].values)
        if 0 < n_perc < len(cols):
            ax.axvline(n_perc - 0.5, color="black", lw=1.2)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([PHYSICS_LABELS.get(pc, pc) for pc in rows], fontsize=10)

        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(v) > 0.55 else "black")

        plt.colorbar(im, ax=ax, label="Mean Spearman ρ (across datasets)")
        ax.set_title(f"Mapping {m} — physics ↔ named audio features",
                     fontsize=12)
        fig.tight_layout()
        fname = f"correlation_handcrafted_{m}.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_learned_bars(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return

    mappings = [m for m in MAPPING_ORDER if m in df["mapping"].values]
    spaces   = [s for s in LEARNED_SPACES if s in df["feature_space"].values]
    metrics  = [pc for pc in PHYSICS_COLS if pc in df["physics_metric"].values]
    if not mappings or not spaces or not metrics:
        return

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(max(12, n_met * 4.5), 5),
                             dpi=DPI, sharey=True)
    if n_met == 1:
        axes = [axes]

    for ax, pc in zip(axes, metrics):
        s_sub = df[df["physics_metric"] == pc]
        x = np.arange(len(spaces))
        width = 0.78 / max(len(mappings), 1)

        for i, m in enumerate(mappings):
            vals = []
            for s in spaces:
                cell = s_sub[(s_sub["mapping"] == m) & (s_sub["feature_space"] == s)]
                vals.append(cell["best_abs_rho"].values[0] if not cell.empty else np.nan)
            offset = (i - len(mappings) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=f"Mapping {m}",
                   color=MAPPING_COLORS.get(m, "grey"),
                   edgecolor="black", linewidth=0.4, alpha=0.92)
            for j, v in enumerate(vals):
                if np.isfinite(v) and v >= 0.02:
                    ax.text(x[j] + offset, v + 0.015, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS.get(s, s) for s in spaces], fontsize=10)
        ax.set_title(PHYSICS_LABELS.get(pc, pc), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Best |Spearman ρ| across axes\n(averaged over datasets)",
                       fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(mappings),
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Best axis-wise correlation in learned feature spaces",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_learned.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: correlation_learned.png")


def plot_param_baselines(df: pd.DataFrame, out_dir: str):
    """One bar plot per FM parameter baseline (pitch, harm_ratio, mod_index)."""
    if df.empty:
        return

    for param in PARAM_BASELINE_SPACES:
        sub = df[df["feature_space"] == param]
        if sub.empty:
            continue

        mappings = [m for m in MAPPING_ORDER if m in sub["mapping"].values]
        metrics  = [pc for pc in PHYSICS_COLS if pc in sub["physics_metric"].values]
        if not mappings or not metrics:
            continue

        param_label = FEATURE_LABELS.get(param, param)
        fig, ax = plt.subplots(figsize=(max(7, len(metrics) * 2.2), 4), dpi=DPI)
        x = np.arange(len(metrics))
        width = 0.78 / max(len(mappings), 1)

        for i, m in enumerate(mappings):
            vals = []
            for pc in metrics:
                cell = sub[(sub["mapping"] == m) & (sub["physics_metric"] == pc)]
                vals.append(cell["mean_rho"].values[0] if not cell.empty else np.nan)
            offset = (i - len(mappings) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=f"Mapping {m}",
                   color=MAPPING_COLORS.get(m, "grey"),
                   edgecolor="black", linewidth=0.4, alpha=0.92)
            for j, v in enumerate(vals):
                if np.isfinite(v) and abs(v) >= 0.02:
                    y_label = v + 0.02 if v >= 0 else v - 0.02
                    va = "bottom" if v >= 0 else "top"
                    ax.text(x[j] + offset, y_label, f"{v:+.2f}",
                            ha="center", va=va, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([PHYSICS_LABELS.get(pc, pc) for pc in metrics], fontsize=10)
        ax.set_ylabel(f"Mean Spearman ρ\n({param_label} ↔ physics, per dataset)",
                      fontsize=10)
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
        ax.set_title(f"{param_label} baseline", fontsize=12)
        fig.tight_layout()
        fname = f"correlation_{param}_baseline.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_event_alignment(all_align: pd.DataFrame, out_dir: str):
    if all_align.empty:
        return

    mappings = sorted(all_align["mapping"].unique())

    pair_df = (all_align.groupby(["physics_metric", "audio_feature"])
                          .size().reset_index().drop(0, axis=1, errors="ignore"))
    pair_labels = [f"{r.physics_metric}\n→ {r.audio_feature}"
                   for _, r in pair_df.iterrows()]
    n_pairs = len(pair_df)
    n_maps  = len(mappings)

    real_mat   = np.full((n_maps, n_pairs), np.nan)
    shuf_mat   = np.full((n_maps, n_pairs), np.nan)
    excess_mat = np.full((n_maps, n_pairs), np.nan)

    for i, m in enumerate(mappings):
        sub = all_align[all_align["mapping"] == m]
        for j, (_, row) in enumerate(pair_df.iterrows()):
            mask = ((sub["physics_metric"] == row["physics_metric"]) &
                    (sub["audio_feature"]  == row["audio_feature"]))
            if mask.any():
                real_mat[i, j]   = sub.loc[mask, "rho"].mean()
                shuf_mat[i, j]   = sub.loc[mask, "rho_shuffled"].mean()
                excess_mat[i, j] = sub.loc[mask, "rho_excess"].mean()

    fig, ax = plt.subplots(figsize=(max(9, n_pairs * 1.5), max(3, n_maps * 1.3)),
                           dpi=DPI)
    im = ax.imshow(excess_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_yticks(range(n_maps))
    ax.set_yticklabels([f"Mapping {m}" for m in mappings])
    for i in range(n_maps):
        for j in range(n_pairs):
            r, s, e = real_mat[i, j], shuf_mat[i, j], excess_mat[i, j]
            if np.isfinite(e):
                txt = f"real {r:+.2f}\nshuf {s:+.2f}\nΔ {e:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(e) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Mean Δρ  (real − shuffled)")
    ax.set_title("Auditory event alignment: real vs. time-shuffled baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "event_alignment.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved: event_alignment.png")


def plot_event_example(data_by_mapping: dict, all_physics: pd.DataFrame,
                       out_dir: str):
    if "magnet_flips" not in all_physics.columns:
        return
    if not data_by_mapping:
        return

    valid = {}
    for m, data in data_by_mapping.items():
        params_df = data.get("params_df")
        named = data.get("named", {})
        perc_df = named.get("perceptual")
        if params_df is None or perc_df is None:
            continue
        if "dataset" not in params_df.columns:
            continue
        if "roughness" not in perc_df.columns:
            continue
        valid[m] = (params_df, perc_df)

    if not valid:
        return

    first_params = next(iter(valid.values()))[0]
    datasets = list(first_params["dataset"].unique())

    for ds in datasets:
        phys_sub = (all_physics[all_physics["dataset"] == ds]
                    .sort_values("time").reset_index(drop=True))
        if len(phys_sub) < 4:
            continue

        mapping_series = {}
        for m, (params_df, perc_df) in valid.items():
            perc_sub = (perc_df[perc_df["dataset"] == ds]
                        .sort_values("time").reset_index(drop=True))
            if len(perc_sub) < 4:
                continue
            joined = phys_sub[["time", "magnet_flips"]].merge(
                perc_sub[["time", "roughness"]], on="time", how="inner"
            ).sort_values("time")
            if len(joined) < 4:
                continue
            mapping_series[m] = joined

        if not mapping_series:
            continue

        first_joined = next(iter(mapping_series.values()))
        t_top = first_joined["time"].values
        flips = first_joined["magnet_flips"].values

        n_rows = 1 + len(mapping_series)
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(12, 2.2 * n_rows), dpi=DPI, sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        axes[0].plot(t_top, flips, color="steelblue", lw=0.8)
        axes[0].set_ylabel("Magnet flips\n(physics)")
        axes[0].set_title(f"Event alignment — {ds}")
        axes[0].grid(alpha=0.3)

        for ax, (m, joined) in zip(axes[1:], mapping_series.items()):
            ax.plot(joined["time"].values, joined["roughness"].values,
                    color="indianred", lw=0.8)
            ax.set_ylabel(f"Roughness\nMapping {m}")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        fname = f"event_alignment_example_{ds}.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")
        break


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+", choices=MAPPINGS, default=MAPPINGS,
                    help="which mappings to evaluate")
    ap.add_argument("--n_shuffles", type=int, default=20,
                    help="permutations for the event-alignment null (default 20)")
    args = ap.parse_args()

    out_dir = get_output_dir()
    print(f"Output directory: {out_dir}")

    print("\nLoading manifest and physics metrics...")
    datasets, metrics_root = load_manifest()
    all_physics = load_all_physics(datasets, metrics_root)
    print(f"  {len(all_physics)} rows from {all_physics['dataset'].nunique()} datasets")

    all_handcrafted = []
    all_learned     = []
    all_align_dfs   = []
    data_by_mapping = {}

    for mapping in args.mapping:
        print(f"\n{'='*70}")
        print(f"  MAPPING {mapping}")
        print(f"{'='*70}")

        paths = get_mapping_paths(mapping)
        data  = load_mapping_data(paths, all_physics)
        if data is None:
            continue

        data_by_mapping[mapping] = data
        print(f"  {len(data['params_df'])} rows | "
              f"named: {list(data['named'].keys())} | "
              f"learned: {list(data['learned'].keys())}")

        print("\n── 1a. Handcrafted features: per-pair Spearman ρ ─────────────")
        df_hc = run_handcrafted_correlation(data, mapping)
        if not df_hc.empty:
            # quick textual summary: strongest |ρ| audio feature per physics metric
            hc_only = df_hc[df_hc["feature_space"].isin(HANDCRAFTED_SPACES)].copy()
            if not hc_only.empty:
                hc_only["abs_rho"] = hc_only["mean_rho"].abs()
                top = (hc_only.sort_values("abs_rho", ascending=False)
                              .groupby("physics_metric")
                              .head(1)[["physics_metric", "audio_feature", "mean_rho"]])
                print("  Strongest correlation per physics metric:")
                for _, row in top.iterrows():
                    print(f"    {row['physics_metric']:<22} ← {row['audio_feature']:<22} "
                          f"ρ = {row['mean_rho']:+.3f}")

            # parameter baselines: routing-fidelity summary
            param_only = df_hc[df_hc["feature_space"].isin(PARAM_BASELINE_SPACES)]
            if not param_only.empty:
                print("  Parameter baselines (routing fidelity):")
                for param in PARAM_BASELINE_SPACES:
                    p_sub = param_only[param_only["feature_space"] == param]
                    if p_sub.empty:
                        continue
                    p_sub = p_sub.copy()
                    p_sub["abs_rho"] = p_sub["mean_rho"].abs()
                    best = p_sub.sort_values("abs_rho", ascending=False).iloc[0]
                    print(f"    {FEATURE_LABELS.get(param, param):<12} "
                          f"strongest: {best['physics_metric']:<22} "
                          f"ρ = {best['mean_rho']:+.3f}")
        all_handcrafted.append(df_hc)

        print("\n── 1b. Learned features: best |ρ| across axes ────────────────")
        df_lr = run_learned_correlation(data, mapping)
        if not df_lr.empty:
            print(df_lr.to_string(index=False))
        all_learned.append(df_lr)

        print(f"\n── 2. Auditory event alignment (n_shuffles={args.n_shuffles}) ──")
        align_df = run_event_alignment(data, all_physics, mapping,
                                        n_shuffles=args.n_shuffles)
        if not align_df.empty:
            summary = (align_df
                       .groupby(["physics_metric", "audio_feature"])
                       .agg(rho_mean=("rho", "mean"),
                            rho_std=("rho", "std"),
                            shuf_mean=("rho_shuffled", "mean"),
                            excess_mean=("rho_excess", "mean"),
                            n_datasets=("rho", "count"))
                       .reset_index())
            print(summary.to_string(index=False))
        all_align_dfs.append(align_df)

    print("\n── Figures and CSVs ──────────────────────────────────────────────")
    df_hc = pd.concat(all_handcrafted, ignore_index=True) if all_handcrafted else pd.DataFrame()
    df_lr = pd.concat(all_learned,     ignore_index=True) if all_learned     else pd.DataFrame()
    df_al = pd.concat(all_align_dfs,   ignore_index=True) if all_align_dfs   else pd.DataFrame()

    plot_handcrafted_heatmaps(df_hc, out_dir)
    plot_learned_bars(df_lr, out_dir)
    plot_param_baselines(df_hc, out_dir)
    plot_event_alignment(df_al, out_dir)
    plot_event_example(data_by_mapping, all_physics, out_dir)

    if not df_hc.empty:
        df_hc.to_csv(os.path.join(out_dir, "evaluation_correlation_handcrafted.csv"),
                     index=False)
        print("  Saved: evaluation_correlation_handcrafted.csv")
    if not df_lr.empty:
        df_lr.to_csv(os.path.join(out_dir, "evaluation_correlation_learned.csv"),
                     index=False)
        print("  Saved: evaluation_correlation_learned.csv")
    if not df_al.empty:
        df_al.to_csv(os.path.join(out_dir, "evaluation_alignment.csv"), index=False)
        print("  Saved: evaluation_alignment.csv")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()