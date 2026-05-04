"""
build_mapped_params.py
======================
Convert ASI simulation metrics into FM-synth parameter CSVs that plug
directly into the SynthMaps pipeline.

For each selected dataset × each mapping (A, B), writes one CSV to
  <synthmapspath>/mapped_params/<mapping>/<dataset_name>.csv

Each CSV has the columns expected by FmSynthDataset:
  time, freq, harm_ratio, mod_index

plus columns used for PCA coloring:
  x (= time, normalised 0–1)
  dataset, mapping  (for multi-dataset PCA later)

The triplet
-----------
dipolar_energy    — total dipolar interaction energy (negative, decreases
                    as the system orders). Smooth, physically central.
                    Drives pitch.
magnet_flips      — number of individual spins that flipped sign since the
                    previous timestep. Bursty: zero most of the time with
                    occasional avalanches up to several hundred flips.
                    Drives modulation depth.
total_mag_angle   — direction of net magnetisation [-π, π]. Circular,
                    settles into a few preferred directions per dataset.
                    Drives harmonic ratio (timbre).

Two mappings — same metric→parameter assignment, different shaping
------------------------------------------------------------------
A — Naive linear baseline
    dipolar_energy   → pitch       linear (inverted) → MIDI 38–86
    total_mag_angle  → harm_ratio  wrap into [0, 2π] → linear
    magnet_flips     → mod_index   linear, per-dataset normalised by max

B — Perceptually shaped
    dipolar_energy   → pitch       same as A (already smooth, no shaping)
    total_mag_angle  → harm_ratio  same as A (already smooth and bounded)
    magnet_flips     → mod_index   log1p (compresses heavy tail of
                                   avalanche events; sparse zeros stay
                                   at MOD_LOW)

A and B share two of three transforms because dipolar_energy and
total_mag_angle are already well-distributed and don't benefit from shaping.
The audible difference between A and B is therefore *entirely* due to the
log1p on magnet_flips, which isolates the effect of compression on the
modulation parameter.

Usage
-----
    python build_mapped_params.py [--outdir <path>] [--mapping A B]
                                  [--subsample N]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from utils import get_path, midi2frequency

# ── FM parameter ranges (must match the SynthMaps grid) ─────────────────────
MIDI_LOW  = 38.0   # MIDI note 38  ≈ 233 Hz
MIDI_HIGH = 86.0   # MIDI note 86  ≈ 1047 Hz
HARM_LOW  = 0.0
HARM_HIGH = 10.0
MOD_LOW   = 0.0
MOD_HIGH  = 10.0


# ── helpers ──────────────────────────────────────────────────────────────────

def scale_linear(v: np.ndarray, out_low: float, out_high: float,
                 v_min: float = None, v_max: float = None) -> np.ndarray:
    """Linearly map v from [v_min, v_max] → [out_low, out_high]."""
    v_min = float(np.min(v)) if v_min is None else float(v_min)
    v_max = float(np.max(v)) if v_max is None else float(v_max)
    if v_max == v_min:
        return np.full_like(v, (out_low + out_high) / 2, dtype=float)
    return out_low + (v - v_min) / (v_max - v_min) * (out_high - out_low)


def scale_log1p(v: np.ndarray, v_max: float,
                out_low: float, out_high: float) -> np.ndarray:
    """log1p(v) / log1p(v_max) → [out_low, out_high].
    Compresses heavy right tail; preserves 0 → out_low."""
    v_clipped = np.clip(v, 0, None)
    log_max = np.log1p(v_max) if v_max > 0 else 1.0
    normed = np.log1p(v_clipped) / log_max
    normed = np.clip(normed, 0, 1)
    return out_low + normed * (out_high - out_low)


def wrap_angle_0_2pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle from [-π, π] (or any range) into [0, 2π].
    The discontinuity at the wraparound is preserved — it represents a
    real fast direction-flip in the macroscopic magnetisation."""
    return np.mod(angle, 2 * np.pi)


def energy_to_freq(energy: np.ndarray,
                   energy_min: float,
                   energy_range: float) -> np.ndarray:
    """Map dipolar_energy to Hz via MIDI.
    Energy is negative and decreases as the system orders, so we invert:
    lower (more negative) energy → higher pitch. Uses per-dataset
    energy_range from the manifest for consistent normalisation."""
    if energy_range == 0:
        midi = np.full(len(energy), (MIDI_LOW + MIDI_HIGH) / 2)
    else:
        normed = (energy - energy_min) / energy_range          # 0 = min, 1 = max
        midi = MIDI_HIGH - normed * (MIDI_HIGH - MIDI_LOW)     # invert direction
    return midi2frequency(midi.astype(np.float64))


# ── mapping functions ────────────────────────────────────────────────────────

def _common(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull the three triplet metrics + time as numpy arrays."""
    return (
        df["dipolar_energy"].values,
        df["total_mag_angle"].values,
        df["magnet_flips"].values,
        df["time"].values,
    )


def mapping_A(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Naive linear baseline — every metric mapped without shaping."""
    energy, angle, flips, t = _common(df)

    freq          = energy_to_freq(energy,
                                   energy_min=energy.min(),
                                   energy_range=meta["energy_range"])
    angle_wrapped = wrap_angle_0_2pi(angle)
    harm_ratio    = scale_linear(angle_wrapped, HARM_LOW, HARM_HIGH,
                                 v_min=0.0, v_max=2 * np.pi)
    flips_max     = float(flips.max()) if flips.max() > 0 else 1.0
    mod_index     = scale_linear(flips, MOD_LOW, MOD_HIGH,
                                 v_min=0.0, v_max=flips_max)
    x             = scale_linear(t.astype(float), 0.0, 1.0)

    return pd.DataFrame({
        "time": t, "freq": freq, "harm_ratio": harm_ratio, "mod_index": mod_index,
        "x": x, "dataset": meta["name"], "mapping": "A",
    })


def mapping_B(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Perceptually shaped — magnet_flips through log1p so avalanche events
    don't blow out the modulation depth and small flips remain audible.
    Energy and angle stay linear since they're already smooth and well-
    distributed."""
    energy, angle, flips, t = _common(df)

    freq          = energy_to_freq(energy,
                                   energy_min=energy.min(),
                                   energy_range=meta["energy_range"])
    angle_wrapped = wrap_angle_0_2pi(angle)
    harm_ratio    = scale_linear(angle_wrapped, HARM_LOW, HARM_HIGH,
                                 v_min=0.0, v_max=2 * np.pi)
    flips_max     = float(flips.max()) if flips.max() > 0 else 1.0
    mod_index     = scale_log1p(flips, v_max=flips_max,
                                out_low=MOD_LOW, out_high=MOD_HIGH)
    x             = scale_linear(t.astype(float), 0.0, 1.0)

    return pd.DataFrame({
        "time": t, "freq": freq, "harm_ratio": harm_ratio, "mod_index": mod_index,
        "x": x, "dataset": meta["name"], "mapping": "B",
    })


MAPPINGS = {"A": mapping_A, "B": mapping_B}


# ── validation ───────────────────────────────────────────────────────────────

def validate_params(out: pd.DataFrame, name: str, mapping: str):
    """Warn if any parameter is outside the expected FM range, then clip."""
    freq_lo = float(midi2frequency(np.array([MIDI_LOW])))
    freq_hi = float(midi2frequency(np.array([MIDI_HIGH])))
    checks = {
        "freq":       (freq_lo, freq_hi),
        "harm_ratio": (HARM_LOW, HARM_HIGH),
        "mod_index":  (MOD_LOW, MOD_HIGH),
    }
    ok = True
    for col, (lo, hi) in checks.items():
        v = out[col].values
        if np.any(v < lo - 1e-6) or np.any(v > hi + 1e-6):
            pct_lo = float(np.mean(v < lo) * 100)
            pct_hi = float(np.mean(v > hi) * 100)
            print(f"  [warn] {name} / mapping {mapping} / {col}: "
                  f"{pct_lo:.1f}% below {lo:.2f}, {pct_hi:.1f}% above {hi:.2f}")
            ok = False
    if not ok:
        print(f"         (values clipped before saving)")
        for col, (lo, hi) in checks.items():
            out[col] = out[col].clip(lo, hi)


def print_param_summary(results: dict):
    """Pooled per-mapping statistics so transforms can be sanity-checked."""
    print("\n── Parameter summary (pooled across all datasets) ──────────────────")
    print(f"  {'mapping':<10}{'param':<14}{'min':>8}{'mean':>8}{'max':>8}{'std':>8}")
    for mapping_name, dfs in sorted(results.items()):
        if not dfs:
            continue
        pooled = pd.concat(dfs, ignore_index=True)
        for col in ("freq", "harm_ratio", "mod_index"):
            v = pooled[col]
            print(f"  {mapping_name:<10}{col:<14}"
                  f"{v.min():8.3f}{v.mean():8.3f}{v.max():8.3f}{v.std():8.3f}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["dipolar_energy", "total_mag_angle", "magnet_flips", "time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="root output dir (default: <synthmapspath>/mapped_params)")
    ap.add_argument("--mapping", nargs="+", choices=["A", "B"],
                    default=["A", "B"])
    ap.add_argument("--subsample", type=int, default=None,
                    help="max timesteps per dataset, evenly spaced. "
                         "Datasets shorter than this are used in full.")
    args = ap.parse_args()

    synthmaps_root = get_path("synthmapspath")
    selected_path  = get_path("selecteddatasetspath")
    metrics_root   = os.path.dirname(get_path("metricspath"))
    out_root = args.outdir or os.path.join(synthmaps_root, "mapped_params")
    os.makedirs(out_root, exist_ok=True)

    with open(selected_path, "r") as f:
        manifest = json.load(f)
    datasets = manifest["datasets"]
    print(f"Manifest: {len(datasets)} datasets, mappings: {args.mapping}"
          + (f", subsample={args.subsample}" if args.subsample else ""))

    results = {m: [] for m in args.mapping}
    skipped = []

    for meta in datasets:
        name = meta["name"]
        csv_path = os.path.join(metrics_root, meta["metrics_csv"])
        if not os.path.exists(csv_path):
            print(f"[skip] missing CSV: {csv_path}")
            skipped.append(name)
            continue

        df = pd.read_csv(csv_path)
        n_original = len(df)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[skip] {name}: missing columns {missing}")
            skipped.append(name)
            continue

        if args.subsample and n_original > args.subsample:
            idx = np.linspace(0, n_original - 1, args.subsample).astype(int)
            df = df.iloc[idx].reset_index(drop=True)
            print(f"  {name}: {n_original} → {len(df)} rows (subsampled)")

        for m in args.mapping:
            out = MAPPINGS[m](df, meta)
            validate_params(out, name, m)

            m_dir = os.path.join(out_root, f"mapping_{m}")
            os.makedirs(m_dir, exist_ok=True)
            out_path = os.path.join(m_dir, f"{name}.csv")
            out.to_csv(out_path, index=True)
            results[m].append(out)

    if skipped:
        print(f"\nSkipped {len(skipped)} datasets: {skipped}")

    print_param_summary(results)

    for m, dfs in results.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(out_root, f"mapping_{m}", "_all_datasets.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Mapping {m}: {len(dfs)} datasets → "
              f"{os.path.join(out_root, f'mapping_{m}')}/")

    print("\nDone.")


if __name__ == "__main__":
    main()