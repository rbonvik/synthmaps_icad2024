"""
classify_simulation.py
======================
Can a classifier recover the simulation model/encoder identity from the
audio features produced by a given mapping? If yes, the sonification has
preserved enough information that the *kind of physics* is audible.

Setup
-----
- One training sample = a non-overlapping window of N consecutive timesteps
  from a single dataset's audio features, summarised by per-feature
  (mean, std).
- Label = simulation model, parsed from dataset names (<id>_<model>_<encoder>)
  or supplied via JSON.
- Cross-validation = GroupKFold with dataset as the group, so test windows
  always come from datasets the classifier has never seen.
- One classifier per feature space, all using logistic regression with L2
  regularisation, standardised input, and balanced class weights.

Feature spaces evaluated (in display order):
    params       — raw FM synth parameters (freq, harm_ratio, mod_index)
                   BASELINE: if this scores as well as the audio spaces,
                   then the labels are already encoded in the mapped
                   parameters and the audio pipeline isn't doing the work.
    handcrafted  — perceptual + spectral features
    mel          — mel spectrogram (time-averaged)
    encodec      — EnCodec embedding
    clap         — CLAP embedding

Metrics
-------
- Accuracy and macro-F1 (cross-validated).
- Per-class precision/recall via held-out predictions.
- Confusion matrix.
- A chance baseline (stratified DummyClassifier) for every setup.
- Per-feature coefficient magnitudes for the handcrafted and params classifiers.

Outputs (under <synthmapspath>/figures/classification/)
-------------------------------------------------------
    classify_summary.png                              — accuracy + macro-F1.
    classify_confusion_<mapping>_<feature_space>.png  — confusion matrix per setup.
    classify_coefs_<mapping>_<space>.png              — feature importance (params, handcrafted).
    evaluation_classification.csv                     — per-(mapping, space) metrics.
    evaluation_classification_per_fold.csv            — per-fold metrics.
    evaluation_classification_reports.txt             — sklearn classification_report.

Usage
-----
    python classify_simulation.py
    python classify_simulation.py --mapping A B --window 25 --stride 25
    python classify_simulation.py --labels /path/to/dataset_labels.json
    python classify_simulation.py --no-params-baseline   # audio only, original behaviour
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_path

# ── config ────────────────────────────────────────────────────────────────────

# FM synth input parameters — the baseline feature space. If a classifier
# trained on these alone recovers the simulation model, the labels are
# already separable in the mapped-parameter space and the audio features
# can only inherit (not add to) that separability.
PARAMS_COLS = ["freq", "harm_ratio", "mod_index"]

PERCEPTUAL_COLS = [
    "hardness", "depth", "brightness", "roughness",
    "warmth", "sharpness", "boominess",
]
SPECTRAL_COLS = [
    "spectral_centroid", "spectral_crest", "spectral_decrease", "spectral_energy",
    "spectral_flatness", "spectral_kurtosis", "spectral_roll_off",
    "spectral_skewness", "spectral_slope", "spectral_spread", "inharmonicity",
]

HANDCRAFTED_SPACES = ["perceptual", "spectral"]
LEARNED_SPACES     = ["mel", "encodec", "clap"]
# Display order: baseline first, then handcrafted, then learned.
ALL_SPACES         = ["params", "handcrafted"] + LEARNED_SPACES

FEATURE_LABELS = {
    "params":      "FM params (baseline)",
    "handcrafted": "Handcrafted (perceptual + spectral)",
    "mel":         "Mel spec.",
    "encodec":     "EnCodec",
    "clap":        "CLAP",
}

MAPPINGS = ["A", "B", "C", "D", "E", "F"]
MAPPING_COLORS = {
    "A": "#2b6cb0",
    "B": "#c2410c",
    "C": "#16a34a",
    "D": "#9333ea",
    "E": "#0891b2",
    "F": "#dc2626",
}

DPI = 200
RANDOM_SEED = 42
N_SPLITS_DEFAULT = 5


# ── small helpers ─────────────────────────────────────────────────────────────

def _sanitise(arr: np.ndarray) -> np.ndarray:
    """
    Replace +/-inf with NaN so a downstream imputer can treat them uniformly.
    The classifier pipeline includes a median SimpleImputer that handles NaN.
    """
    arr = np.asarray(arr, dtype=float)
    return np.where(np.isinf(arr), np.nan, arr)


def _window_summary(chunk: np.ndarray) -> np.ndarray:
    """Per-column (nanmean, nanstd) of a window, with NaN/inf zeroed out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu  = np.nanmean(chunk, axis=0)
        sig = np.nanstd(chunk, axis=0)
    mu  = np.nan_to_num(mu,  nan=0.0, posinf=0.0, neginf=0.0)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([mu, sig])


# ── path helpers ──────────────────────────────────────────────────────────────

def get_mapping_paths(mapping: str) -> dict:
    root = get_path("synthmapspath")
    out  = os.path.join(root, "results", f"mapping_{mapping}")
    return {
        "params_csv":     os.path.join(root, "mapped_params",
                                       f"mapping_{mapping}", "_all_datasets.csv"),
        "perceptual_csv": os.path.join(out, "fm_synth_perceptual_features.csv"),
        "spectral_csv":   os.path.join(out, "fm_synth_spectral_features.csv"),
        "mel_npy":        os.path.join(out, "fm_synth_mel_spectrograms_mean.npy"),
        "encodec_npy":    os.path.join(out, "fm_synth_encodec_embeddings.npy"),
        "clap_npy":       os.path.join(out, "fm_synth_clap_embeddings.npy"),
    }


def get_output_dir() -> str:
    d = os.path.join(get_path("synthmapspath"), "figures", "classification")
    os.makedirs(d, exist_ok=True)
    return d


# ── label parsing ─────────────────────────────────────────────────────────────

def label_from_name(dataset_name: str, target: str) -> Optional[str]:
    """
    Parse the simulation model and/or encoder out of '<id>_<model>_<encoder>'.
    Returns None if the name doesn't match.
    """
    parts = dataset_name.split("_")
    if len(parts) < 3:
        return None
    model, encoder = parts[-2], parts[-1]
    if target == "model":
        return model
    if target == "encoder":
        return encoder
    if target == "both":
        return f"{model}_{encoder}"
    raise ValueError(f"Unknown target: {target!r}")


def load_labels(labels_path: Optional[str], target: str,
                dataset_names: list[str]) -> dict:
    """Build {dataset_name: label} from JSON if given, otherwise parse names."""
    if labels_path is not None:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label file not found: {labels_path}")
        with open(labels_path) as f:
            labels = json.load(f)
        print(f"  Loaded {len(labels)} dataset labels from {labels_path}")
        return labels

    labels = {}
    n_unparsed = 0
    for name in dataset_names:
        lab = label_from_name(name, target)
        if lab is None:
            n_unparsed += 1
        else:
            labels[name] = lab
    if n_unparsed:
        print(f"  [warn] {n_unparsed} dataset names did not match "
              f"<id>_<model>_<encoder> — those datasets will be dropped")
    print(f"  Derived {len(labels)} labels (target={target!r})")
    return labels


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class MappingData:
    params_df:   pd.DataFrame
    params_arr:  np.ndarray            # (N, D_params) — FM synth inputs
    params_cols: list[str]
    handcrafted: np.ndarray            # (N, D_hc)
    hc_cols:     list[str]
    learned:     dict[str, np.ndarray] # space → (N, D)


def _load_feature_csv(path: str, expected_cols: list[str], n_rows: int
                      ) -> tuple[np.ndarray, list[str]]:
    """Load a feature CSV and return (array aligned to n_rows, used columns)."""
    if not os.path.exists(path):
        return np.empty((n_rows, 0)), []
    raw = pd.read_csv(path, index_col=0)
    used = [c for c in expected_cols if c in raw.columns]
    if not used:
        return np.empty((n_rows, 0)), []
    aligned = raw[used].reindex(range(n_rows))
    return aligned.values.astype(float), used


def _extract_params(params_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Pull the FM synth input parameters out of the per-row mapping CSV.
    These are the actual inputs to the synthesizer — if they're separable
    by class, the audio downstream can only inherit that separability.
    """
    used = [c for c in PARAMS_COLS if c in params_df.columns]
    if not used:
        return np.empty((len(params_df), 0)), []
    arr = params_df[used].values.astype(float)
    return _sanitise(arr), used


def load_mapping_features(paths: dict) -> Optional[MappingData]:
    if not os.path.exists(paths["params_csv"]):
        print(f"  [skip] params CSV not found: {paths['params_csv']}")
        return None

    params_df = pd.read_csv(paths["params_csv"]).reset_index(drop=True)
    if "dataset" not in params_df.columns or "time" not in params_df.columns:
        print("  [skip] params CSV missing 'dataset' or 'time' column")
        return None
    N = len(params_df)

    # Baseline feature space: the raw FM synth inputs themselves.
    params_arr, params_cols = _extract_params(params_df)
    if not params_cols:
        print(f"  [warn] no FM parameter columns found in {paths['params_csv']} "
              f"(expected any of {PARAMS_COLS}) — params baseline will be skipped")

    # handcrafted = perceptual + spectral, side by side, sanitised once
    perc_arr, perc_cols = _load_feature_csv(paths["perceptual_csv"],
                                            PERCEPTUAL_COLS, N)
    spec_arr, spec_cols = _load_feature_csv(paths["spectral_csv"],
                                            SPECTRAL_COLS, N)
    if perc_cols and spec_cols:
        hc_arr  = np.concatenate([perc_arr, spec_arr], axis=1)
        hc_cols = perc_cols + spec_cols
    elif perc_cols:
        hc_arr, hc_cols = perc_arr, perc_cols
    elif spec_cols:
        hc_arr, hc_cols = spec_arr, spec_cols
    else:
        hc_arr, hc_cols = np.empty((N, 0)), []
    hc_arr = _sanitise(hc_arr)

    learned: dict[str, np.ndarray] = {}
    if os.path.exists(paths["mel_npy"]):
        learned["mel"] = _sanitise(np.load(paths["mel_npy"])[:N])
    if os.path.exists(paths["encodec_npy"]):
        embs = np.load(paths["encodec_npy"])[:N]
        learned["encodec"] = _sanitise(embs.reshape(len(embs), -1))
    if os.path.exists(paths["clap_npy"]):
        learned["clap"] = _sanitise(np.load(paths["clap_npy"])[:N])

    return MappingData(params_df=params_df,
                       params_arr=params_arr, params_cols=params_cols,
                       handcrafted=hc_arr, hc_cols=hc_cols,
                       learned=learned)


# ── windowing ─────────────────────────────────────────────────────────────────

def build_windows(values: np.ndarray, dataset_ids: np.ndarray,
                  times: np.ndarray, window: int, stride: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build strided per-dataset windows. Returns ((W, 2D) summary, (W,) groups).
    Each row is concatenated nanmean and nanstd of the window.
    """
    chunks, groups = [], []
    for ds in pd.unique(dataset_ids):
        idx = np.where(dataset_ids == ds)[0]
        if len(idx) < window:
            continue
        idx = idx[np.argsort(times[idx], kind="stable")]
        vals = values[idx]
        for start in range(0, len(vals) - window + 1, stride):
            chunks.append(_window_summary(vals[start:start + window]))
            groups.append(ds)

    if not chunks:
        return np.empty((0, values.shape[1] * 2)), np.array([])
    return np.stack(chunks), np.array(groups)


# ── classifier pipeline ───────────────────────────────────────────────────────

def make_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """
    SimpleImputer (median) → StandardScaler → LogisticRegression.

    The imputer is the safety net: any NaN that survived sanitisation (e.g. an
    all-NaN window column) gets filled with the fold-train median, which is also
    the cross-validation-correct way to impute.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, class_weight="balanced",
            random_state=seed,
        )),
    ])


def cross_validated_classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             n_splits: int = N_SPLITS_DEFAULT,
                             seed: int = RANDOM_SEED) -> Optional[dict]:
    """
    GroupKFold cross-validation. Returns metrics + held-out predictions + averaged
    feature coefficients, or None if the data is too small to cross-validate.
    """
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    classes = np.array(sorted(np.unique(y)))

    per_fold, y_true_all, y_pred_all, coefs_list = [], [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue

        pipe = make_pipeline(seed=seed).fit(X[tr], y[tr])
        y_pred = pipe.predict(X[te])
        acc = accuracy_score(y[te], y_pred)
        f1  = f1_score(y[te], y_pred, average="macro", zero_division=0)

        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        dummy.fit(X[tr], y[tr])
        y_dum = dummy.predict(X[te])
        acc_d = accuracy_score(y[te], y_dum)
        f1_d  = f1_score(y[te], y_dum, average="macro", zero_division=0)

        per_fold.append((acc, f1, acc_d, f1_d))
        y_true_all.append(y[te])
        y_pred_all.append(y_pred)

        lr = pipe.named_steps["clf"]
        if len(lr.classes_) == len(classes) and np.all(lr.classes_ == classes):
            coefs_list.append(lr.coef_)

    if not per_fold:
        return None

    arr = np.array(per_fold)
    coefs_mean = (np.mean(coefs_list, axis=0)
                  if len(coefs_list) == len(per_fold) else None)

    return {
        "acc_mean":   float(arr[:, 0].mean()),
        "acc_std":    float(arr[:, 0].std()),
        "f1_mean":    float(arr[:, 1].mean()),
        "f1_std":     float(arr[:, 1].std()),
        "acc_chance": float(arr[:, 2].mean()),
        "f1_chance":  float(arr[:, 3].mean()),
        "per_fold":   per_fold,
        "y_true":     np.concatenate(y_true_all),
        "y_pred":     np.concatenate(y_pred_all),
        "classes":    classes,
        "coefs":      coefs_mean,
        "n_samples":  len(X),
        "n_groups":   n_groups,
        "n_splits":   splits,
    }


# ── feature-set preparation ───────────────────────────────────────────────────

def _filter_to_labeled(data: MappingData, labels: dict
                       ) -> Optional[tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]]:
    """
    Drop rows whose dataset isn't in `labels`. Returns
    (dataset_ids, times, has_label_mask, the labels array). None if nothing left.
    """
    dataset_ids = data.params_df["dataset"].values
    times       = data.params_df["time"].values
    label_arr   = np.array([labels.get(ds) for ds in dataset_ids])
    has_label   = np.array([lab is not None for lab in label_arr])

    n_unlabeled = int((~has_label).sum())
    if n_unlabeled:
        unlabeled_ds = sorted(set(dataset_ids[~has_label]))
        head = unlabeled_ds[:5]
        ellipsis = "..." if len(unlabeled_ds) > 5 else ""
        print(f"  [info] {n_unlabeled} rows from {len(unlabeled_ds)} unlabeled "
              f"datasets dropped: {head}{ellipsis}")

    if has_label.sum() == 0:
        return None
    return dataset_ids[has_label], times[has_label], has_label, label_arr[has_label]


def prepare_feature_sets(data: MappingData, has_label: np.ndarray,
                         include_params: bool = True
                         ) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Return {feature_space: (values, feature_names)} for the rows we'll classify.
    Sanitisation (inf→NaN) was already done at load time; we don't re-impute
    here because the sklearn pipeline does that fold-correctly.

    When `include_params` is True, the FM synth input parameters are included
    as the "params" feature space. This is the baseline: if it scores as well
    as the audio spaces, the labels are already separable in the mapped
    parameters and the audio pipeline isn't doing the discriminative work.
    """
    feature_sets: dict[str, tuple[np.ndarray, list[str]]] = {}

    if include_params and data.params_cols:
        feature_sets["params"] = (data.params_arr[has_label], data.params_cols)

    if data.hc_cols:
        feature_sets["handcrafted"] = (data.handcrafted[has_label], data.hc_cols)

    for space, arr in data.learned.items():
        feature_sets[space] = (
            arr[has_label],
            [f"{space}_{i}" for i in range(arr.shape[1])],
        )
    return feature_sets


# ── one mapping ───────────────────────────────────────────────────────────────

def run_classification_for_mapping(data: MappingData, labels: dict, mapping: str,
                                   window: int, stride: int,
                                   include_params: bool = True
                                   ) -> tuple[dict, list, list]:
    """
    Per feature space: build windows, train+CV, collect metrics.
    Returns (detailed_results, summary_rows, per_fold_rows).
    """
    filtered = _filter_to_labeled(data, labels)
    if filtered is None:
        print("  [skip] no rows match any label")
        return {}, [], []
    dataset_ids, times, has_label, _ = filtered

    feature_sets = prepare_feature_sets(data, has_label,
                                        include_params=include_params)

    results:  dict[str, dict] = {}
    summary:  list[dict] = []
    per_fold: list[dict] = []

    for fs_name, (X, fnames) in feature_sets.items():
        X_win, g_win = build_windows(X, dataset_ids, times, window, stride)
        if len(X_win) == 0:
            print(f"  [skip] {fs_name}: no windows "
                  f"(datasets shorter than window={window}?)")
            continue

        y_win = np.array([labels[g] for g in g_win])
        if len(np.unique(y_win)) < 2:
            print(f"  [skip] {fs_name}: only one class present")
            continue
        if len(np.unique(g_win)) < 2:
            print(f"  [skip] {fs_name}: only one dataset has windows")
            continue

        res = cross_validated_classify(X_win, y_win, g_win)
        if res is None:
            print(f"  [skip] {fs_name}: no usable folds")
            continue

        # build_windows stacks mean then std, so names are doubled accordingly.
        res["feature_names"] = (
            [f"{c}_mean" for c in fnames] + [f"{c}_std" for c in fnames]
        )
        results[fs_name] = res

        print(f"  {FEATURE_LABELS.get(fs_name, fs_name):<36}  "
              f"acc {res['acc_mean']:.3f}±{res['acc_std']:.3f}  "
              f"(chance {res['acc_chance']:.3f})   "
              f"macro-F1 {res['f1_mean']:.3f}±{res['f1_std']:.3f}  "
              f"[{res['n_samples']} windows, {res['n_groups']} datasets, "
              f"{res['n_splits']} folds]")

        summary.append({
            "mapping":       mapping,
            "feature_space": fs_name,
            "n_windows":     res["n_samples"],
            "n_datasets":    res["n_groups"],
            "n_classes":     len(res["classes"]),
            "acc_mean":      res["acc_mean"],
            "acc_std":       res["acc_std"],
            "acc_chance":    res["acc_chance"],
            "f1_mean":       res["f1_mean"],
            "f1_std":        res["f1_std"],
            "f1_chance":     res["f1_chance"],
        })

        for fold_i, (a, f, ad, fd) in enumerate(res["per_fold"]):
            per_fold.append({
                "mapping":       mapping,
                "feature_space": fs_name,
                "fold":          fold_i,
                "acc":           a,
                "f1":            f,
                "acc_chance":    ad,
                "f1_chance":     fd,
            })

    return results, summary, per_fold


# ── plotting helpers ──────────────────────────────────────────────────────────

def _grouped_bars(ax: plt.Axes, summary_df: pd.DataFrame, spaces: list[str],
                  mappings: list[str], mean_col: str, std_col: str,
                  chance_col: str, title: str):
    """Side-by-side bars over `spaces`, one bar per mapping. Adds chance ticks."""
    x = np.arange(len(spaces))
    width = 0.78 / max(len(mappings), 1)
    chance_marks: list[tuple[float, float, float]] = []

    for i, m in enumerate(mappings):
        offset = (i - len(mappings) / 2 + 0.5) * width
        vals, stds, chances = [], [], []
        for s in spaces:
            cell = summary_df[(summary_df["mapping"] == m) &
                              (summary_df["feature_space"] == s)]
            if cell.empty:
                vals.append(np.nan); stds.append(np.nan); chances.append(np.nan)
            else:
                vals.append(float(cell[mean_col].values[0]))
                stds.append(float(cell[std_col].values[0]))
                chances.append(float(cell[chance_col].values[0]))

        ax.bar(x + offset, vals, width, yerr=stds, capsize=2,
               label=f"Mapping {m}",
               color=MAPPING_COLORS.get(m, "grey"),
               edgecolor="black", linewidth=0.4, alpha=0.92,
               error_kw=dict(ecolor="black", lw=0.6, alpha=0.55))
        for j, v in enumerate(vals):
            if np.isfinite(v) and v >= 0.02:
                err = stds[j] if np.isfinite(stds[j]) else 0
                ax.text(x[j] + offset, v + err + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        for j, c in enumerate(chances):
            if np.isfinite(c):
                chance_marks.append((x[j] + offset, c, width))

    for xc, c, w in chance_marks:
        ax.hlines(c, xc - w / 2, xc + w / 2,
                  colors="black", linestyles="--", lw=0.9, alpha=0.7)

    # Visually separate the params baseline from the audio spaces with a divider.
    if "params" in spaces and len(spaces) > 1:
        sep_x = spaces.index("params") + 0.5
        ax.axvline(sep_x, color="grey", linestyle=":", lw=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS.get(s, s) for s in spaces],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_summary(summary_df: pd.DataFrame, out_dir: str):
    if summary_df.empty:
        return

    mappings = sorted(summary_df["mapping"].unique())
    available = set(summary_df["feature_space"].values)
    spaces = [s for s in ALL_SPACES if s in available]

    n_map, n_sp = len(mappings), len(spaces)
    fig, axes = plt.subplots(1, 2, figsize=(max(11, n_sp * n_map * 0.9 + 4), 5),
                             dpi=DPI)

    _grouped_bars(axes[0], summary_df, spaces, mappings,
                  "acc_mean", "acc_std", "acc_chance", "Accuracy")
    _grouped_bars(axes[1], summary_df, spaces, mappings,
                  "f1_mean", "f1_std", "f1_chance", "Macro F1")

    chance_handle = Line2D([0], [0], color="black", ls="--", lw=0.9,
                           label="Chance (stratified)")
    handles, lbls = axes[0].get_legend_handles_labels()
    handles.append(chance_handle); lbls.append("Chance (stratified)")
    fig.legend(handles, lbls, loc="lower center", ncol=min(n_map + 1, 6),
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Simulation-model classification from sonified audio\n"
                 "(FM params is the baseline: if it matches the audio spaces, "
                 "labels are encoded in the mapping itself)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "classify_summary.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved: classify_summary.png")


def plot_confusion(res: dict, mapping: str, fs_name: str, out_dir: str):
    classes = res["classes"]
    cm = confusion_matrix(res["y_true"], res["y_pred"],
                          labels=classes, normalize="true")
    fig, ax = plt.subplots(
        figsize=(max(4, len(classes) * 0.9 + 1),
                 max(3.5, len(classes) * 0.7 + 1.5)),
        dpi=DPI,
    )
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Row-normalised count")
    ax.set_title(f"Confusion: mapping {mapping}, "
                 f"{FEATURE_LABELS.get(fs_name, fs_name)}\n"
                 f"acc {res['acc_mean']:.3f}, macro-F1 {res['f1_mean']:.3f}")
    fig.tight_layout()
    fname = f"classify_confusion_{mapping}_{fs_name}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_coefs(res: dict, mapping: str, space: str, out_dir: str,
               top_k: int = 15):
    """
    Plot mean |coefficient| for an interpretable feature space (params
    or handcrafted). Skips if coefs are unavailable or the space is too
    high-dimensional to be meaningful (e.g. mel/embeddings).
    """
    if res.get("coefs") is None or res.get("feature_names") is None:
        return
    coefs = np.abs(res["coefs"]).mean(axis=0)
    fnames = np.array(res["feature_names"])

    # For params we show all features (typically 6: 3 means + 3 stds);
    # for handcrafted we cap at top_k.
    n_show = len(fnames) if space == "params" else min(top_k, len(fnames))
    order = np.argsort(coefs)[::-1][:n_show]

    fig, ax = plt.subplots(figsize=(8, max(3, n_show * 0.32)), dpi=DPI)
    y = np.arange(len(order))
    ax.barh(y, coefs[order][::-1], color="#2b6cb0", edgecolor="black", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(fnames[order][::-1], fontsize=9)
    ax.set_xlabel("Mean |coefficient| (across classes, standardised features)")
    title_space = FEATURE_LABELS.get(space, space)
    ax.set_title(f"Feature importance — mapping {mapping}, {title_space}\n"
                 f"(higher = more useful for distinguishing simulation models)")
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    fname = f"classify_coefs_{mapping}_{space}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── output writers ────────────────────────────────────────────────────────────

def write_csv(df: pd.DataFrame, path: str, label: str):
    if df.empty:
        return
    df.to_csv(path, index=False)
    print(f"  Saved: {os.path.basename(path)}  ({label})")


def write_classification_reports(detailed: dict, out_dir: str):
    if not detailed:
        return
    path = os.path.join(out_dir, "evaluation_classification_reports.txt")
    with open(path, "w") as f:
        for (mapping, fs_name), res in detailed.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"Mapping {mapping} | {FEATURE_LABELS.get(fs_name, fs_name)}\n")
            f.write(f"{'='*70}\n")
            f.write(classification_report(res["y_true"], res["y_pred"],
                                          zero_division=0))
    print(f"  Saved: {os.path.basename(path)}")


def print_baseline_comparison(summary_df: pd.DataFrame):
    """
    Compact textual comparison: for each mapping, how does params-only
    accuracy stack up against the audio spaces? If audio - params is
    small or negative, the audio features aren't doing discriminative
    work that the parameters couldn't already do.
    """
    if summary_df.empty or "params" not in summary_df["feature_space"].values:
        return

    print("\n── Baseline comparison: params vs audio ─────────────────────────")
    print("  Δ = (audio acc) − (params acc). Δ ≤ 0 means audio adds nothing.")
    print()
    for mapping in sorted(summary_df["mapping"].unique()):
        sub = summary_df[summary_df["mapping"] == mapping]
        params_row = sub[sub["feature_space"] == "params"]
        if params_row.empty:
            continue
        p_acc = float(params_row["acc_mean"].values[0])
        print(f"  Mapping {mapping}:  params acc = {p_acc:.3f}")
        for s in ["handcrafted", "mel", "encodec", "clap"]:
            row = sub[sub["feature_space"] == s]
            if row.empty:
                continue
            a_acc = float(row["acc_mean"].values[0])
            delta = a_acc - p_acc
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
            print(f"    {FEATURE_LABELS.get(s, s):<36}  "
                  f"acc = {a_acc:.3f}   Δ = {delta:+.3f} {arrow}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+", choices=MAPPINGS, default=MAPPINGS,
                    help="which mappings to evaluate")
    ap.add_argument("--window", type=int, default=25,
                    help="window size in timesteps (default 25)")
    ap.add_argument("--stride", type=int, default=None,
                    help="stride between windows (default = window)")
    ap.add_argument("--target", choices=["model", "encoder", "both"], default="model",
                    help="what to predict, parsed from <id>_<model>_<encoder>")
    ap.add_argument("--labels", type=str, default=None,
                    help="optional JSON {dataset_name: label} that overrides parsing")
    ap.add_argument("--no-params-baseline", action="store_true",
                    help="skip the FM-parameter baseline (audio spaces only). "
                         "Use this only if you've already established that the "
                         "mapping doesn't trivially separate classes.")
    return ap.parse_args()


def main():
    args = parse_args()
    stride = args.stride if args.stride is not None else args.window
    include_params = not args.no_params_baseline

    out_dir = get_output_dir()
    print(f"Output directory: {out_dir}")
    print(f"\nWindow={args.window}, stride={stride}, target={args.target!r}, "
          f"params_baseline={include_params}")

    all_summary:  list[dict] = []
    all_perfold:  list[dict] = []
    detailed:     dict[tuple[str, str], dict] = {}
    labels:       Optional[dict] = None

    for mapping in args.mapping:
        print(f"\n{'='*70}\n  MAPPING {mapping}\n{'='*70}")

        data = load_mapping_features(get_mapping_paths(mapping))
        if data is None:
            continue

        if labels is None:
            dataset_names = sorted(data.params_df["dataset"].unique().tolist())
            print("\nBuilding labels...")
            labels = load_labels(args.labels, args.target, dataset_names)
            print(f"  {len(set(labels.values()))} classes: "
                  f"{sorted(set(labels.values()))}")

        print(f"  {len(data.params_df)} rows | "
              f"params cols: {len(data.params_cols)} | "
              f"handcrafted cols: {len(data.hc_cols)} | "
              f"learned spaces: {list(data.learned.keys())}")

        results, summary, per_fold = run_classification_for_mapping(
            data, labels, mapping, args.window, stride,
            include_params=include_params,
        )
        for fs_name, res in results.items():
            detailed[(mapping, fs_name)] = res
        all_summary.extend(summary)
        all_perfold.extend(per_fold)

    print("\n── Figures and CSVs ──────────────────────────────────────────────")
    summary_df = pd.DataFrame(all_summary)
    perfold_df = pd.DataFrame(all_perfold)

    plot_summary(summary_df, out_dir)
    write_csv(summary_df,
              os.path.join(out_dir, "evaluation_classification.csv"),
              "per-(mapping, feature_space) metrics")
    write_csv(perfold_df,
              os.path.join(out_dir, "evaluation_classification_per_fold.csv"),
              "per-fold metrics")

    for (mapping, fs_name), res in detailed.items():
        plot_confusion(res, mapping, fs_name, out_dir)

    # Coefficient plots for interpretable spaces only.
    for mapping in args.mapping:
        for space in ["params", "handcrafted"]:
            if (mapping, space) in detailed:
                plot_coefs(detailed[(mapping, space)], mapping, space, out_dir)

    write_classification_reports(detailed, out_dir)
    print_baseline_comparison(summary_df)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()