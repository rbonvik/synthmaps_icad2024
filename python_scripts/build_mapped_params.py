"""
build_mapped_params.py
======================
Convert ASI simulation metrics into FM-synth parameter CSVs that plug
directly into the SynthMaps pipeline.

For each selected dataset × each mapping, writes one CSV to
  <synthmapspath>/mapped_params/mapping_<name>/<dataset_name>.csv

Each CSV has the columns expected by FmSynthDataset:
  time, freq, harm_ratio, mod_index

plus columns used for PCA coloring:
  x (= time, normalised 0–1)
  dataset, mapping  (for multi-dataset PCA later)

The triplet of source metrics
-----------------------------
dipolar_energy     — total dipolar interaction energy (negative, decreases
                     as the system orders). Smooth, physically central.
                     We use |E| so that "more ordered" = "higher value",
                     matching the natural direction of the other two metrics.
hamming_from_init  — Hamming distance from the initial configuration.
                     Slow, near-monotonic drift measure.
magnet_flips       — spins flipped since the previous timestep.
                     Bursty: zero most of the time with occasional
                     avalanches up to several hundred flips.

Direction convention
--------------------
All three metrics follow the same rule: high metric value → high
parameter value (no inversion). For dipolar energy this means
high |E| (strongly ordered) → high pitch / harm_ratio / mod_index.
For hamming and flips it means high activity → high parameter value.

Normalisation
-------------
All three metrics use 5th/95th percentile clipping before linear
scaling to [0, 1]. This is consistent with how the SynthMaps
pipeline already treats perceptual and spectral features, and it
handles the heavy-tailed / bimodal distribution of dipolar energy
gracefully.

  dipolar_energy   — global |E| percentiles across all datasets
  hamming_from_init — global percentiles across all datasets
  magnet_flips     — per-dataset percentiles (absolute flip counts
                     depend on system size; what carries the
                     perceptual content is burst shape)

NB: the normalisation policy follows the *metric*, not the
parameter slot, under any assignment.

Mappings
--------
All six mappings use linear transforms; they differ only in which
metric drives which FM parameter slot. This isolates assignment as
the only variable across mappings.

Dipolar energy as the pitch driver
A - DHM — dipolar_energy → pitch, hamming_from_init → harm_ratio, magnet_flips → mod_index
B - DMH — dipolar_energy → pitch, magnet_flips → harm_ratio, hamming_from_init → mod_index

Hamming distance as the pitch driver
C - HDM — hamming_from_init → pitch, dipolar_energy → harm_ratio, magnet_flips → mod_index
D - HMD — hamming_from_init → pitch, magnet_flips → harm_ratio, dipolar_energy → mod_index

Magnet flips as the pitch driver
E - MDH — magnet_flips → pitch, dipolar_energy → harm_ratio, hamming_from_init → mod_index
F - MHD — magnet_flips → pitch, hamming_from_init → harm_ratio, dipolar_energy → mod_index

Usage
-----
    python build_mapped_params.py
    python build_mapped_params.py --mapping A C E
    python build_mapped_params.py --subsample 500
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

# ── normalisation percentiles ───────────────────────────────────────────────
# Clip metric values at these percentiles before linear scaling.
# Matches the SynthMaps pipeline's treatment of perceptual / spectral feats.
CLIP_LO_PCT = 0.05
CLIP_HI_PCT = 0.95


# ── slot helpers ─────────────────────────────────────────────────────────────

def _slot_range(slot: str) -> tuple[float, float]:
    if slot == "pitch":
        return MIDI_LOW, MIDI_HIGH        # special-cased: returned as Hz below
    if slot == "harm_ratio":
        return HARM_LOW, HARM_HIGH
    if slot == "mod_index":
        return MOD_LOW, MOD_HIGH
    raise ValueError(f"unknown slot: {slot}")


def _to_pitch(midi: np.ndarray) -> np.ndarray:
    """Convert a MIDI-valued series to Hz."""
    return midi2frequency(midi.astype(np.float64))


def _to_slot(normed: np.ndarray, slot: str, invert: bool = False) -> np.ndarray:
    """Map a [0, 1]-normalised series to the target slot range.
    If slot == "pitch", returns Hz (via MIDI). If invert, flips the
    direction before scaling."""
    lo, hi = _slot_range(slot)
    if invert:
        normed = 1.0 - normed
    scaled = lo + normed * (hi - lo)
    if slot == "pitch":
        return _to_pitch(scaled)
    return scaled


# ── transform primitives ─────────────────────────────────────────────────────

def scale_linear(v: np.ndarray, out_low: float, out_high: float,
                 v_min: float = None, v_max: float = None) -> np.ndarray:
    """Linearly map v from [v_min, v_max] → [out_low, out_high]."""
    v_min = float(np.min(v)) if v_min is None else float(v_min)
    v_max = float(np.max(v)) if v_max is None else float(v_max)
    if v_max == v_min:
        return np.full_like(v, (out_low + out_high) / 2, dtype=float)
    return out_low + (v - v_min) / (v_max - v_min) * (out_high - out_low)


def _clip_and_norm(values: np.ndarray,
                   lo: float, hi: float) -> np.ndarray:
    """Clip values to [lo, hi], then linearly map to [0, 1].
    Returns 0.5 for degenerate ranges."""
    if hi <= lo:
        return np.full(len(values), 0.5)
    clipped = np.clip(values, lo, hi)
    return (clipped - lo) / (hi - lo)


# ── per-metric transformers ──────────────────────────────────────────────────
#
# Each transformer takes a metric series and a `slot` argument and
# returns a values-in-FM-range numpy array. The per-metric shaping
# and normalisation policy live here, so reassigning a metric to a
# different slot does not change how it is normalised.
#
# All three metrics use the same convention:
#   high metric value (after any preprocessing) → high parameter value.
# No invert flags. This makes mapping comparisons fair: differences
# across A–F come only from which metric drives which slot.
# ─────────────────────────────────────────────────────────────────────────────

def transform_dipolar_energy(values: np.ndarray, slot: str,
                             dipolar_clip: tuple[float, float],
                             **_) -> np.ndarray:
    """Use |E| (binding magnitude), clipped at global 5th/95th percentile,
    then linearly scaled. High |E| (more ordered) → high parameter value."""
    mag = np.abs(values)
    lo, hi = dipolar_clip
    normed = _clip_and_norm(mag, lo, hi)
    return _to_slot(normed, slot)


def transform_hamming_from_init(values: np.ndarray, slot: str,
                                hamming_clip: tuple[float, float],
                                **_) -> np.ndarray:
    """Clipped at global 5th/95th percentile, then linearly scaled.
    High drift → high parameter value."""
    lo, hi = hamming_clip
    normed = _clip_and_norm(values, lo, hi)
    return _to_slot(normed, slot)


def transform_magnet_flips(values: np.ndarray, slot: str, **_) -> np.ndarray:
    """Per-dataset 5th/95th percentile clipping, then linear scaling.
    Per-dataset normalisation is used because absolute flip counts depend
    on system size; what carries the perceptual content is burst shape.
    High flip count → high parameter value."""
    lo = float(np.quantile(values, CLIP_LO_PCT))
    hi = float(np.quantile(values, CLIP_HI_PCT))
    normed = _clip_and_norm(values, lo, hi)
    return _to_slot(normed, slot)


# ── mapping registry ─────────────────────────────────────────────────────────
#
# A mapping is fully described by:
#   assignment   — dict mapping metric_name → slot ("pitch" | "harm_ratio" | "mod_index")
#   transforms   — dict mapping metric_name → transformer function
#
# Since all six mappings use the same (linear) transforms, the only
# thing that varies is the assignment dict.
#
# Slots in each assignment must form a permutation of
# {"pitch", "harm_ratio", "mod_index"}.
# ─────────────────────────────────────────────────────────────────────────────

LINEAR_TRANSFORMS = {
    "dipolar_energy":    transform_dipolar_energy,
    "hamming_from_init": transform_hamming_from_init,
    "magnet_flips":      transform_magnet_flips,
}


def _assignment(pitch: str, harm: str, mod: str) -> dict[str, str]:
    """Build a metric→slot assignment dict from the three driving metrics."""
    return {pitch: "pitch", harm: "harm_ratio", mod: "mod_index"}


MAPPINGS: dict[str, dict] = {
    # Dipolar energy as pitch driver
    "A": {
        "assignment": _assignment("dipolar_energy", "hamming_from_init", "magnet_flips"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "dipolar_energy→pitch, hamming→harm_ratio, flips→mod_index",
    },
    "B": {
        "assignment": _assignment("dipolar_energy", "magnet_flips", "hamming_from_init"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "dipolar_energy→pitch, flips→harm_ratio, hamming→mod_index",
    },
    # Hamming distance as pitch driver
    "C": {
        "assignment": _assignment("hamming_from_init", "dipolar_energy", "magnet_flips"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "hamming→pitch, dipolar_energy→harm_ratio, flips→mod_index",
    },
    "D": {
        "assignment": _assignment("hamming_from_init", "magnet_flips", "dipolar_energy"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "hamming→pitch, flips→harm_ratio, dipolar_energy→mod_index",
    },
    # Magnet flips as pitch driver
    "E": {
        "assignment": _assignment("magnet_flips", "dipolar_energy", "hamming_from_init"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "flips→pitch, dipolar_energy→harm_ratio, hamming→mod_index",
    },
    "F": {
        "assignment": _assignment("magnet_flips", "hamming_from_init", "dipolar_energy"),
        "transforms": LINEAR_TRANSFORMS,
        "description": "flips→pitch, hamming→harm_ratio, dipolar_energy→mod_index",
    },
}


def _validate_assignment(name: str, assignment: dict[str, str]):
    slots = sorted(assignment.values())
    if slots != sorted(["pitch", "harm_ratio", "mod_index"]):
        raise ValueError(
            f"mapping {name}: assignment must be a bijection over "
            f"{{pitch, harm_ratio, mod_index}}, got slots {slots}"
        )
    metrics = sorted(assignment.keys())
    if metrics != sorted(["dipolar_energy", "hamming_from_init", "magnet_flips"]):
        raise ValueError(
            f"mapping {name}: assignment must cover all three metrics, "
            f"got {metrics}"
        )


def apply_mapping(name: str, df: pd.DataFrame, meta: dict,
                  dipolar_clip: tuple[float, float],
                  hamming_clip: tuple[float, float]) -> pd.DataFrame:
    """Apply a registered mapping to a single dataset."""
    spec = MAPPINGS[name]
    _validate_assignment(name, spec["assignment"])

    t = df["time"].values
    out_cols: dict[str, np.ndarray] = {}

    for metric, slot in spec["assignment"].items():
        values = df[metric].values
        transformer = spec["transforms"][metric]
        out_cols[slot] = transformer(
            values, slot=slot, meta=meta,
            dipolar_clip=dipolar_clip,
            hamming_clip=hamming_clip,
        )

    return pd.DataFrame({
        "time":       t,
        "freq":       out_cols["pitch"],
        "harm_ratio": out_cols["harm_ratio"],
        "mod_index":  out_cols["mod_index"],
        "x":          scale_linear(t.astype(float), 0.0, 1.0),
        "dataset":    meta["name"],
        "mapping":    name,
    })


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
    print(f"  {'mapping':<12}{'param':<14}{'min':>8}{'mean':>8}{'max':>8}{'std':>8}")
    for mapping_name, dfs in sorted(results.items()):
        if not dfs:
            continue
        pooled = pd.concat(dfs, ignore_index=True)
        for col in ("freq", "harm_ratio", "mod_index"):
            v = pooled[col]
            print(f"  {mapping_name:<12}{col:<14}"
                  f"{v.min():8.3f}{v.mean():8.3f}{v.max():8.3f}{v.std():8.3f}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["dipolar_energy", "hamming_from_init", "magnet_flips", "time"]


def compute_global_quantiles(datasets: list, metrics_root: str,
                             column: str,
                             transform=None) -> tuple[float, float]:
    """Pool a column across all selected datasets and return its
    (5th, 95th) percentiles. If `transform` is given, it is applied
    to each dataset's values before pooling (e.g. np.abs)."""
    pooled = []
    for meta in datasets:
        csv_path = os.path.join(metrics_root, meta["metrics_csv"])
        if not os.path.exists(csv_path):
            continue
        try:
            col = pd.read_csv(csv_path, usecols=[column])[column].values
            if transform is not None:
                col = transform(col)
            pooled.append(col)
        except (ValueError, KeyError):
            continue
    if not pooled:
        return 0.0, 0.0
    all_vals = np.concatenate(pooled)
    lo = float(np.quantile(all_vals, CLIP_LO_PCT))
    hi = float(np.quantile(all_vals, CLIP_HI_PCT))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="root output dir (default: <synthmapspath>/mapped_params)")
    ap.add_argument("--mapping", nargs="+",
                    choices=list(MAPPINGS.keys()),
                    default=list(MAPPINGS.keys()),
                    help="which mappings to build (default: all)")
    ap.add_argument("--subsample", type=int, default=None,
                    help="max timesteps per dataset, evenly spaced. "
                         "Datasets shorter than this are used in full.")
    args = ap.parse_args()

    synthmaps_root = get_path("synthmapspath")
    selected_path  = get_path("datasetSummaryPath")
    metrics_root   = os.path.dirname(get_path("metricspath"))
    out_root = args.outdir or os.path.join(synthmaps_root, "mapped_params")
    os.makedirs(out_root, exist_ok=True)

    with open(selected_path, "r") as f:
        manifest = json.load(f)
    datasets = manifest["datasets"]
    print(f"Manifest: {len(datasets)} datasets")
    print(f"Building mappings: {args.mapping}")
    for m in args.mapping:
        print(f"  {m}: {MAPPINGS[m]['description']}")
    if args.subsample:
        print(f"Subsample: {args.subsample} timesteps per dataset")

    # Global percentile ranges (computed once across the whole manifest)
    dipolar_clip = compute_global_quantiles(
        datasets, metrics_root, "dipolar_energy", transform=np.abs)
    hamming_clip = compute_global_quantiles(
        datasets, metrics_root, "hamming_from_init")

    print(f"Global |dipolar_energy| {int(CLIP_LO_PCT*100)}th/{int(CLIP_HI_PCT*100)}th "
          f"percentile: [{dipolar_clip[0]:.2f}, {dipolar_clip[1]:.2f}]")
    print(f"Global hamming_from_init {int(CLIP_LO_PCT*100)}th/{int(CLIP_HI_PCT*100)}th "
          f"percentile: [{hamming_clip[0]:.2f}, {hamming_clip[1]:.2f}]")
    if dipolar_clip[1] <= dipolar_clip[0]:
        print("  [warn] dipolar clip range is degenerate — "
              "dipolar-driven slot will be constant.")
    if hamming_clip[1] <= hamming_clip[0]:
        print("  [warn] hamming clip range is degenerate — "
              "hamming-driven slot will be constant.")

    results: dict[str, list[pd.DataFrame]] = {m: [] for m in args.mapping}
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
            out = apply_mapping(
                m, df, meta,
                dipolar_clip=dipolar_clip,
                hamming_clip=hamming_clip,
            )
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