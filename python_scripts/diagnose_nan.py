"""
diagnose_nans.py
================
Locate NaN / inf values in the feature files for one mapping.

Run:
    python diagnose_nans.py --mapping B
    python diagnose_nans.py --mapping A B C D E F   # all of them

For each mapping, prints:
  - which file(s) contain NaN or inf
  - how many bad rows
  - which columns (handcrafted) or column indices (learned) are affected
  - a few example rows so you can cross-reference with params_df (dataset + time)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_path

PERCEPTUAL_COLS = [
    "hardness", "depth", "brightness", "roughness", "warmth", "sharpness", "boominess"
]
SPECTRAL_COLS = [
    "spectral_centroid", "spectral_crest", "spectral_decrease", "spectral_energy",
    "spectral_flatness", "spectral_kurtosis", "spectral_roll_off", "spectral_skewness",
    "spectral_slope", "spectral_spread", "inharmonicity",
]


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


def _summarise_array(name: str, arr: np.ndarray, params_df: pd.DataFrame | None,
                     col_names: list[str] | None = None):
    """Report NaN/inf locations in a (N, D) array."""
    print(f"\n── {name} ── shape {arr.shape}, dtype {arr.dtype}")
    if arr.size == 0:
        print("  (empty)")
        return

    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"  total NaN: {n_nan}    total inf: {n_inf}")

    if n_nan == 0 and n_inf == 0:
        print("  → clean")
        return

    # per-column counts
    col_nan = np.isnan(arr).sum(axis=0)
    col_inf = np.isinf(arr).sum(axis=0)
    bad_cols = np.where((col_nan + col_inf) > 0)[0]

    print(f"  columns with NaN or inf: {len(bad_cols)} of {arr.shape[1]}")
    for j in bad_cols[:20]:
        label = col_names[j] if col_names is not None and j < len(col_names) else f"col[{j}]"
        print(f"    {label:<30}  NaN={int(col_nan[j]):>6}  inf={int(col_inf[j]):>6}")
    if len(bad_cols) > 20:
        print(f"    ... ({len(bad_cols) - 20} more)")

    # which rows
    row_bad = (~np.isfinite(arr)).any(axis=1)
    n_bad_rows = int(row_bad.sum())
    print(f"  rows with any NaN/inf: {n_bad_rows} of {arr.shape[0]} "
          f"({100*n_bad_rows/arr.shape[0]:.2f}%)")

    if params_df is not None and n_bad_rows > 0 and len(params_df) == arr.shape[0]:
        bad_idx = np.where(row_bad)[0]
        sample = bad_idx[:8]
        print("  example bad rows (index → dataset, time):")
        for i in sample:
            ds = params_df.iloc[i].get("dataset", "?")
            t  = params_df.iloc[i].get("time", "?")
            print(f"    row {i}: dataset={ds}, time={t}")

        # which datasets are affected
        if "dataset" in params_df.columns:
            affected = params_df.iloc[bad_idx]["dataset"].value_counts()
            print(f"  datasets affected: {len(affected)} of "
                  f"{params_df['dataset'].nunique()}")
            print("  top 10 by bad-row count:")
            for ds, n in affected.head(10).items():
                print(f"    {ds:<40}  {n} bad rows")


def diagnose_mapping(mapping: str):
    print("\n" + "=" * 70)
    print(f"  MAPPING {mapping}")
    print("=" * 70)
    paths = get_mapping_paths(mapping)

    # params first — we use its dataset/time columns to locate offending rows
    if not os.path.exists(paths["params_csv"]):
        print(f"  [skip] params CSV not found: {paths['params_csv']}")
        return
    params_df = pd.read_csv(paths["params_csv"]).reset_index(drop=True)
    print(f"\n── params CSV ── {len(params_df)} rows")
    print(f"  columns: {list(params_df.columns)}")
    # check params themselves for NaN/inf
    num_cols = params_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        sub = params_df[num_cols].values.astype(float)
        n_nan = int(np.isnan(sub).sum())
        n_inf = int(np.isinf(sub).sum())
        print(f"  params NaN: {n_nan}   inf: {n_inf}")

    N = len(params_df)

    # perceptual CSV
    if os.path.exists(paths["perceptual_csv"]):
        raw = pd.read_csv(paths["perceptual_csv"], index_col=0)
        avail = [c for c in PERCEPTUAL_COLS if c in raw.columns]
        print(f"\n  perceptual file rows: {len(raw)}  (params has {N})")
        if len(raw) != N:
            print(f"  [warn] row count mismatch! perceptual has {len(raw)}, params has {N}")
        arr = raw[avail].reindex(range(N)).values.astype(float)
        _summarise_array("perceptual features", arr, params_df, avail)
    else:
        print(f"\n  [missing] {paths['perceptual_csv']}")

    # spectral CSV
    if os.path.exists(paths["spectral_csv"]):
        raw = pd.read_csv(paths["spectral_csv"], index_col=0)
        avail = [c for c in SPECTRAL_COLS if c in raw.columns]
        print(f"\n  spectral file rows: {len(raw)}  (params has {N})")
        if len(raw) != N:
            print(f"  [warn] row count mismatch! spectral has {len(raw)}, params has {N}")
        arr = raw[avail].reindex(range(N)).values.astype(float)
        _summarise_array("spectral features", arr, params_df, avail)
    else:
        print(f"\n  [missing] {paths['spectral_csv']}")

    # learned spaces
    for label, path_key in [("mel spectrograms", "mel_npy"),
                            ("EnCodec embeddings", "encodec_npy"),
                            ("CLAP embeddings", "clap_npy")]:
        path = paths[path_key]
        if not os.path.exists(path):
            print(f"\n  [missing] {path}")
            continue
        arr = np.load(path)
        if arr.ndim > 2:
            arr_flat = arr.reshape(arr.shape[0], -1)
        else:
            arr_flat = arr
        print(f"\n  {label} file shape: {arr.shape}  (params has {N})")
        if arr.shape[0] != N:
            print(f"  [warn] row count mismatch!")
        arr_flat = arr_flat[:N]
        _summarise_array(label, arr_flat, params_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+",
                    choices=["A", "B", "C", "D", "E", "F"],
                    default=["B"],
                    help="which mapping(s) to inspect")
    args = ap.parse_args()
    for m in args.mapping:
        diagnose_mapping(m)


if __name__ == "__main__":
    main()