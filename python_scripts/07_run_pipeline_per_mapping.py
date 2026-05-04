"""
07_run_pipeline_per_mapping.py
==============================
Orchestrates the SynthMaps feature-extraction and PCA pipeline (scripts 02–06)
for each of the FM-parameter mappings (A, B) produced by build_mapped_params.py.

For each mapping the script:
  1. Points the pipeline at the mapping's parameter CSV
     (<synthmapspath>/mapped_params/mapping_X/_all_datasets.csv)
  2. Renders one concatenated WAV per dataset    (NEW)
  3. Runs timbral feature extraction  (≈ script 02)
  4. Runs spectral feature extraction (≈ script 03)
  5. Renders mel spectrograms         (≈ script 04)
  6. Renders EnCodec + CLAP embeddings (≈ script 05)
  7. Renders PCA plots                (≈ script 06)
  All outputs go to <synthmapspath>/results/mapping_X/

paths.json must contain:
  synthmapspath          — root SynthMaps data directory

Usage
-----
    # all mappings, 4 workers, default 50 ms per timestep for audio
    python 07_run_pipeline_per_mapping.py --n_jobs 4

    # single mapping with longer per-step audio (1 second per row)
    python 07_run_pipeline_per_mapping.py --mapping A --audio_dur 1.0

    # only render audio, skip features and PCA
    python 07_run_pipeline_per_mapping.py --mapping A \
        --skip perceptual spectral mel embeddings pca

    # skip audio rendering (the original behaviour)
    python 07_run_pipeline_per_mapping.py --skip audio

    # skip embeddings (no GPU / no internet)
    python 07_run_pipeline_per_mapping.py --skip embeddings

Audio output
------------
For each dataset in the manifest, one WAV is written to:
    <synthmapspath>/results/mapping_X/audio/<dataset_name>.wav
Each WAV is the concatenation of all that dataset's timesteps, each
synthesised for `--audio_dur` seconds (default 0.05).
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import get_path, frequency2midi
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ── constants matching the SynthMaps pipeline ─────────────────────────────────

SR   = 48000
DUR  = 1.0          # seconds per timestep for perceptual/spectral/mel
DUR_EMBED = 0.25    # seconds per timestep for embeddings (as in script 05)
N_MELS = 200

MAPPINGS = ["A", "B"]

STAGES = ["audio", "perceptual", "spectral", "mel", "embeddings", "pca"]


# ── path helpers ──────────────────────────────────────────────────────────────

def get_mapping_paths(mapping: str) -> dict:
    """Return all relevant paths for one mapping."""
    synthmaps_root = get_path("synthmapspath")
    params_csv = os.path.join(
        synthmaps_root, "mapped_params", f"mapping_{mapping}", "_all_datasets.csv"
    )
    out_dir = os.path.join(synthmaps_root, "results", f"mapping_{mapping}")
    fig_dir = os.path.join(synthmaps_root, "figures", f"mapping_{mapping}")
    audio_dir = os.path.join(out_dir, "audio")
    return {
        "params_csv":   params_csv,
        "out_dir":      out_dir,
        "fig_dir":      fig_dir,
        "audio_dir":    audio_dir,
        "perceptual_json": os.path.join(out_dir, "fm_synth_perceptual_features.json"),
        "perceptual_csv":  os.path.join(out_dir, "fm_synth_perceptual_features.csv"),
        "spectral_json":   os.path.join(out_dir, "fm_synth_spectral_features.json"),
        "spectral_csv":    os.path.join(out_dir, "fm_synth_spectral_features.csv"),
        "mel_npy":         os.path.join(out_dir, "fm_synth_mel_spectrograms_mean.npy"),
        "encodec_npy":     os.path.join(out_dir, "fm_synth_encodec_embeddings.npy"),
        "clap_npy":        os.path.join(out_dir, "fm_synth_clap_embeddings.npy"),
    }


def check_params_csv(paths: dict, mapping: str) -> bool:
    p = paths["params_csv"]
    if not os.path.exists(p):
        print(f"  [ERROR] params CSV not found for mapping {mapping}: {p}")
        print(f"          Run build_mapped_params.py first.")
        return False
    df = pd.read_csv(p)
    n = len(df)
    print(f"  Params CSV: {n} rows  ({p})")
    return True


# ── FM synthesis (shared across stages) ──────────────────────────────────────

def load_params(params_csv: str) -> pd.DataFrame:
    df = pd.read_csv(params_csv)
    required = {"freq", "harm_ratio", "mod_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"params CSV missing columns: {missing}")
    return df


def synth_row(row, sr: int, dur: float):
    """Synthesise one row of FM params. Returns (audio, freq, harm_ratio, mod_index)."""
    from utils import fm_synth_gen
    f  = np.array([row.freq])
    hr = np.array([row.harm_ratio])
    mi = np.array([row.mod_index])
    audio = fm_synth_gen(int(dur * sr), sr, f, hr, mi)
    return audio, row.freq, row.harm_ratio, row.mod_index


# ── stage 0: audio rendering (one WAV per dataset) ───────────────────────────

def run_audio(paths: dict, audio_dur: float, force: bool = False,
              fade_ms: float = 2.0):
    """Render one concatenated WAV per dataset.

    Each timestep contributes `audio_dur` seconds of FM-synth audio. Adjacent
    timesteps are crossfaded by `fade_ms` to suppress clicks at parameter
    transitions (cheap raised-cosine fade). The final WAV is the concatenation
    of all that dataset's timesteps in order.
    """
    from utils import fm_synth_gen
    from scipy.io import wavfile

    os.makedirs(paths["audio_dir"], exist_ok=True)
    df = load_params(paths["params_csv"])

    if "dataset" not in df.columns:
        print("  [warn] params CSV has no 'dataset' column — "
              "writing a single concatenated WAV for everything")
        df["dataset"] = "all"

    samples_per_step = int(audio_dur * SR)
    if samples_per_step < 2:
        raise ValueError(f"audio_dur {audio_dur}s is too short (yields "
                         f"{samples_per_step} samples per step)")

    # crossfade window — clipped if longer than half the per-step audio
    fade_samples = min(int(fade_ms * 1e-3 * SR), samples_per_step // 2)
    if fade_samples > 0:
        # raised cosine, applied at both head and tail of each step
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
        fade_in  = np.concatenate([ramp, np.ones(samples_per_step - fade_samples)])
        fade_out = np.concatenate([np.ones(samples_per_step - fade_samples), ramp[::-1]])
        envelope = fade_in * fade_out
    else:
        envelope = np.ones(samples_per_step)

    datasets = df["dataset"].unique()
    print(f"  Rendering {len(datasets)} datasets × {samples_per_step} samples/step "
          f"({audio_dur}s/step) → {paths['audio_dir']}")

    for ds_name in tqdm(datasets, desc="  audio"):
        out_wav = os.path.join(paths["audio_dir"], f"{ds_name}.wav")
        if os.path.exists(out_wav) and not force:
            continue

        sub = df[df["dataset"] == ds_name]
        n_steps = len(sub)
        full = np.zeros(n_steps * samples_per_step, dtype=np.float32)

        for i, row in enumerate(sub.itertuples(index=False)):
            chunk = fm_synth_gen(samples_per_step, SR,
                                 np.array([row.freq]),
                                 np.array([row.harm_ratio]),
                                 np.array([row.mod_index]))
            full[i * samples_per_step : (i + 1) * samples_per_step] = chunk * envelope

        # normalise to peak −3 dBFS to leave headroom; protect against silence
        peak = float(np.max(np.abs(full)))
        if peak > 0:
            full = full * (0.707 / peak)

        # int16 PCM is widely compatible
        wavfile.write(out_wav, SR, (full * 32767.0).astype(np.int16))

    print(f"  Saved {len(datasets)} WAV files → {paths['audio_dir']}")


# ── stage 1: perceptual features ──────────────────────────────────────────────

def _extract_perceptual(args):
    i, row_dict, sr = args
    from utils import fm_synth_gen
    import timbral_models
    row = type("R", (), row_dict)()
    audio, freq, ratio, index = synth_row(row, sr, DUR)
    return {
        "index":              i,
        "freq":               freq,
        "harm_ratio":         ratio,
        "mod_index":          index,
        "hardness":           timbral_models.timbral_hardness(audio, fs=sr),
        "depth":              timbral_models.timbral_depth(audio, fs=sr),
        "brightness":         timbral_models.timbral_brightness(audio, fs=sr),
        "roughness":          timbral_models.timbral_roughness(audio, fs=sr),
        "warmth":             timbral_models.timbral_warmth(audio, fs=sr),
        "sharpness":          timbral_models.timbral_sharpness(audio, fs=sr),
        "boominess":          timbral_models.timbral_booming(audio, fs=sr),
    }


def run_perceptual(paths: dict, n_jobs: int):
    out_json = paths["perceptual_json"]
    out_csv  = paths["perceptual_csv"]
    if os.path.exists(out_csv):
        print("  [skip] perceptual features already exist")
        return

    df = load_params(paths["params_csv"])
    print(f"  Extracting perceptual features for {len(df)} rows "
          f"with {n_jobs} worker(s)...")

    args = [(i, row._asdict(), SR) for i, row in enumerate(df.itertuples(index=False))]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_extract_perceptual, a): a[0] for a in args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  perceptual"):
            results.append(fut.result())

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df_out = pd.DataFrame(results).set_index("index").sort_index()
    df_out.to_csv(out_csv)
    print(f"  Saved perceptual features → {out_csv}")


# ── stage 2: spectral features ────────────────────────────────────────────────

def _extract_spectral(args):
    i, row_dict, sr = args
    from utils import fm_synth_gen, frequency2midi
    from pytimbre.waveform import Waveform
    from pytimbre.spectral.spectra import SpectrumByFFT
    row = type("R", (), row_dict)()
    audio, freq, ratio, index = synth_row(row, sr, DUR)
    wfm      = Waveform(audio, sr, 0.0)
    spectrum = SpectrumByFFT(wfm, 4096)
    return {
        "index":              i,
        "freq":               freq,
        "harm_ratio":         ratio,
        "mod_index":          index,
        "spectral_centroid":  spectrum.spectral_centroid,
        "spectral_crest":     spectrum.spectral_crest,
        "spectral_decrease":  spectrum.spectral_decrease,
        "spectral_energy":    spectrum.spectral_energy,
        "spectral_flatness":  spectrum.spectral_flatness,
        "spectral_kurtosis":  spectrum.spectral_kurtosis,
        "spectral_roll_off":  spectrum.spectral_roll_off,
        "spectral_skewness":  spectrum.spectral_skewness,
        "spectral_slope":     spectrum.spectral_slope,
        "spectral_spread":    spectrum.spectral_spread,
        "inharmonicity":      spectrum.inharmonicity,
    }


def run_spectral(paths: dict, n_jobs: int):
    out_json = paths["spectral_json"]
    out_csv  = paths["spectral_csv"]
    if os.path.exists(out_csv):
        print("  [skip] spectral features already exist")
        return

    df = load_params(paths["params_csv"])
    print(f"  Extracting spectral features for {len(df)} rows "
          f"with {n_jobs} worker(s)...")

    args = [(i, row._asdict(), SR) for i, row in enumerate(df.itertuples(index=False))]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_extract_spectral, a): a[0] for a in args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  spectral"):
            results.append(fut.result())

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df_out = pd.DataFrame(results).set_index("index").sort_index()
    df_out.to_csv(out_csv)
    print(f"  Saved spectral features → {out_csv}")


# ── stage 3: mel spectrograms ─────────────────────────────────────────────────

def run_mel(paths: dict):
    out_npy = paths["mel_npy"]
    if os.path.exists(out_npy):
        print("  [skip] mel spectrograms already exist")
        return

    import torch
    from torchaudio.functional import amplitude_to_DB
    from torchaudio.transforms import MelSpectrogram
    from utils import fm_synth_gen

    df = load_params(paths["params_csv"])
    print(f"  Rendering mel spectrograms for {len(df)} rows...")

    mel_spec = MelSpectrogram(
        sample_rate=SR, n_fft=4096, f_min=20, f_max=10000,
        pad=1, n_mels=N_MELS, power=2, norm="slaney", mel_scale="slaney"
    )

    all_mel = np.zeros((len(df), N_MELS))
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df),
                                 desc="  mel")):
        audio = fm_synth_gen(int(DUR * SR), SR,
                             np.array([row.freq]),
                             np.array([row.harm_ratio]),
                             np.array([row.mod_index]))
        mel = mel_spec(torch.tensor(audio, dtype=torch.float32))
        mel_avg = mel.mean(dim=1, keepdim=True)
        mel_avg_db = amplitude_to_DB(mel_avg, multiplier=10,
                                     amin=1e-5, db_multiplier=20, top_db=80)
        all_mel[i] = mel_avg_db.numpy().T

    np.save(out_npy, all_mel)
    print(f"  Saved mel spectrograms → {out_npy}  shape={all_mel.shape}")


# ── stage 4: embeddings (EnCodec + CLAP) ─────────────────────────────────────

def run_embeddings(paths: dict):
    from utils import fm_synth_gen
    from frechet_audio_distance import FrechetAudioDistance

    synthmaps_root = get_path("synthmapspath")
    ckpt_dir       = os.path.join(synthmaps_root, "checkpoints")

    df = load_params(paths["params_csv"])
    n  = len(df)

    samples = int(DUR_EMBED * SR)
    print(f"  Pre-rendering {n} synths at {DUR_EMBED}s...")
    all_audio = np.zeros((n, samples))
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n,
                                 desc="  render")):
        all_audio[i] = fm_synth_gen(samples, SR,
                                    np.array([row.freq]),
                                    np.array([row.harm_ratio]),
                                    np.array([row.mod_index]))

    enc_npy = paths["encodec_npy"]
    if os.path.exists(enc_npy):
        print("  [skip] EnCodec embeddings already exist")
    else:
        print("  Rendering EnCodec embeddings...")
        frechet = FrechetAudioDistance(
            ckpt_dir=os.path.join(ckpt_dir, "encodec"),
            model_name="encodec",
            sample_rate=SR, channels=2, verbose=False,
        )
        test_embs = frechet.get_embeddings([all_audio[0]], SR)
        all_embs  = np.zeros((n, test_embs.shape[0], test_embs.shape[1]))
        for i in tqdm(range(n), desc="  encodec"):
            all_embs[i] = frechet.get_embeddings([all_audio[i]], SR)
        np.save(enc_npy, all_embs)
        print(f"  Saved EnCodec embeddings → {enc_npy}  shape={all_embs.shape}")

    clap_npy = paths["clap_npy"]
    if os.path.exists(clap_npy):
        print("  [skip] CLAP embeddings already exist")
    else:
        print("  Rendering CLAP embeddings...")
        frechet = FrechetAudioDistance(
            ckpt_dir=os.path.join(ckpt_dir, "clap"),
            model_name="clap",
            sample_rate=SR, verbose=False,
        )
        test_embs = frechet.get_embeddings([all_audio[0]], SR)
        all_embs  = np.zeros((n, test_embs.shape[-1]))
        for i in tqdm(range(n), desc="  clap"):
            all_embs[i] = frechet.get_embeddings([all_audio[i]], SR)
        np.save(clap_npy, all_embs)
        print(f"  Saved CLAP embeddings    → {clap_npy}  shape={all_embs.shape}")


# ── stage 5: PCA plots ────────────────────────────────────────────────────────

def array2fluid_dataset(array: np.ndarray) -> dict:
    return {"cols": array.shape[1],
            "data": {str(i): array[i].tolist() for i in range(len(array))}}


def pca2d(X: np.ndarray) -> tuple[np.ndarray, float]:
    pca = PCA(n_components=2, whiten=True, random_state=42)
    result = pca.fit_transform(X)
    return result, float(pca.explained_variance_ratio_.sum())


def run_pca(paths: dict, mapping: str):
    os.makedirs(paths["fig_dir"], exist_ok=True)
    df_params = load_params(paths["params_csv"])

    raw_csv = pd.read_csv(paths["params_csv"])
    if "dataset" in raw_csv.columns and "x" in raw_csv.columns:
        datasets = raw_csv["dataset"].unique()
        n_ds = len(datasets)
        ds_idx = {name: i for i, name in enumerate(datasets)}
        color_max = 0.9
        r = np.array([ds_idx[d] / max(n_ds - 1, 1) for d in raw_csv["dataset"]]) * color_max
        g = raw_csv["x"].values * color_max
        b = np.zeros(len(r))
        alpha = np.full(len(r), 0.3)
        colors = np.stack([r, g, b, alpha], axis=-1)
    else:
        t = np.arange(len(df_params))
        v = (t / t.max() * 0.9)
        colors = np.stack([v, v, v, np.full(len(v), 0.3)], axis=-1)

    fontsize = 14
    dpi = 200

    def scatter(xy, title, evr, out_png, out_json_path):
        fig, ax = plt.subplots(dpi=dpi)
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c=colors)
        ax.set_xlabel("PCA – 1st component", fontsize=fontsize)
        ax.set_ylabel("PCA – 2nd component", fontsize=fontsize)
        ax.set_title(f"{title}\nexplained variance: {evr:.2f}", fontsize=fontsize)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        with open(out_json_path, "w") as f:
            json.dump(array2fluid_dataset(xy), f)

    import matplotlib.pyplot as plt

    scaler = MinMaxScaler()
    midi   = frequency2midi(df_params["freq"].values.astype(np.float64))
    X_params = scaler.fit_transform(
        np.column_stack([midi, df_params["harm_ratio"], df_params["mod_index"]])
    )
    xy, evr = pca2d(X_params)
    scatter(xy, f"PCA of FM params — mapping {mapping}", evr,
            os.path.join(paths["fig_dir"], "pca_params.png"),
            os.path.join(paths["out_dir"], "pca_params.json"))
    print(f"  PCA params done  (EVR={evr:.2f})")

    perc_csv = paths["perceptual_csv"]
    if os.path.exists(perc_csv):
        df_p = pd.read_csv(perc_csv)
        cols = ["hardness", "depth", "brightness", "roughness",
                "warmth", "sharpness", "boominess"]
        df_p = df_p[cols].replace([np.inf, -np.inf], np.nan)
        for c in cols:
            df_p[c] = df_p[c].fillna(df_p[c].max())
        n = 0.1
        for c in cols:
            df_p[c] = df_p[c].clip(df_p[c].quantile(n), df_p[c].quantile(1 - n))
        X = MinMaxScaler().fit_transform(df_p)
        xy, evr = pca2d(X)
        scatter(xy, f"PCA of perceptual features — mapping {mapping}", evr,
                os.path.join(paths["fig_dir"], "pca_perceptual.png"),
                os.path.join(paths["out_dir"], "pca_perceptual.json"))
        print(f"  PCA perceptual done  (EVR={evr:.2f})")

    spec_csv = paths["spectral_csv"]
    if os.path.exists(spec_csv):
        df_s = pd.read_csv(spec_csv)
        cols = ["spectral_centroid", "spectral_crest", "spectral_decrease",
                "spectral_energy", "spectral_flatness", "spectral_kurtosis",
                "spectral_roll_off", "spectral_skewness", "spectral_slope",
                "spectral_spread", "inharmonicity"]
        df_s = df_s[cols]
        for c in ["spectral_roll_off", "spectral_centroid", "spectral_spread"]:
            df_s[c] = frequency2midi(df_s[c].values.astype(np.float64))
        n = 0.1
        for c in cols:
            df_s[c] = df_s[c].clip(df_s[c].quantile(n), df_s[c].quantile(1 - n))
        X = MinMaxScaler().fit_transform(df_s)
        xy, evr = pca2d(X)
        scatter(xy, f"PCA of spectral features — mapping {mapping}", evr,
                os.path.join(paths["fig_dir"], "pca_spectral.png"),
                os.path.join(paths["out_dir"], "pca_spectral.json"))
        print(f"  PCA spectral done  (EVR={evr:.2f})")

    if os.path.exists(paths["mel_npy"]):
        mels = np.load(paths["mel_npy"])
        xy, evr = pca2d(mels)
        scatter(xy, f"PCA of mel spectrograms — mapping {mapping}", evr,
                os.path.join(paths["fig_dir"], "pca_mels_mean.png"),
                os.path.join(paths["out_dir"], "pca_mels_mean.json"))
        print(f"  PCA mel done  (EVR={evr:.2f})")

    if os.path.exists(paths["encodec_npy"]):
        embs = np.load(paths["encodec_npy"])
        embs_2d = embs.reshape(embs.shape[0], -1)
        xy, evr = pca2d(embs_2d)
        scatter(xy, f"PCA of EnCodec embeddings — mapping {mapping}", evr,
                os.path.join(paths["fig_dir"], "pca_encodec.png"),
                os.path.join(paths["out_dir"], "pca_encodec.json"))
        print(f"  PCA encodec done  (EVR={evr:.2f})")

    if os.path.exists(paths["clap_npy"]):
        embs = np.load(paths["clap_npy"])
        xy, evr = pca2d(embs)
        scatter(xy, f"PCA of CLAP embeddings — mapping {mapping}", evr,
                os.path.join(paths["fig_dir"], "pca_clap.png"),
                os.path.join(paths["out_dir"], "pca_clap.json"))
        print(f"  PCA CLAP done  (EVR={evr:.2f})")

    print(f"  All PCA figures → {paths['fig_dir']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", nargs="+", choices=MAPPINGS, default=MAPPINGS,
                    help="which mappings to process (default: all)")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[],
                    help="stages to skip")
    ap.add_argument("--n_jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                    help="parallel workers for perceptual + spectral extraction")
    ap.add_argument("--no_embeddings", action="store_true",
                    help="shorthand for --skip embeddings (use when no GPU)")
    ap.add_argument("--audio_dur", type=float, default=0.05,
                    help="seconds per timestep when rendering listenable audio "
                         "(default 0.05 — a 500-step dataset → ~25s)")
    ap.add_argument("--audio_force", action="store_true",
                    help="re-render audio WAVs even if they already exist")
    args = ap.parse_args()

    skip = set(args.skip)
    if args.no_embeddings:
        skip.add("embeddings")

    print(f"Mappings : {args.mapping}")
    print(f"Stages   : {[s for s in STAGES if s not in skip]}")
    print(f"Workers  : {args.n_jobs}")
    if "audio" not in skip:
        print(f"Audio    : {args.audio_dur}s/step")

    for mapping in args.mapping:
        print(f"\n{'='*60}")
        print(f"  MAPPING {mapping}")
        print(f"{'='*60}")

        paths = get_mapping_paths(mapping)
        os.makedirs(paths["out_dir"], exist_ok=True)
        os.makedirs(paths["fig_dir"], exist_ok=True)

        if not check_params_csv(paths, mapping):
            continue

        if "audio" not in skip:
            print("\n── Stage 0: audio rendering")
            run_audio(paths, audio_dur=args.audio_dur, force=args.audio_force)

        if "perceptual" not in skip:
            print("\n── Stage 1: perceptual features")
            run_perceptual(paths, args.n_jobs)

        if "spectral" not in skip:
            print("\n── Stage 2: spectral features")
            run_spectral(paths, args.n_jobs)

        if "mel" not in skip:
            print("\n── Stage 3: mel spectrograms")
            run_mel(paths)

        if "embeddings" not in skip:
            print("\n── Stage 4: embeddings")
            run_embeddings(paths)

        if "pca" not in skip:
            print("\n── Stage 5: PCA plots")
            run_pca(paths, mapping)

    print("\nAll done.")


if __name__ == "__main__":
    main()