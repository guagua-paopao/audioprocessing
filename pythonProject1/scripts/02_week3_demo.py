"""
Week 3 Demo:
- Read wav (exit if not found)
- Visualize Mel triangular filterbank
- Plot Mel spectrogram (log-mel)
- Compute and display MFCC / Δ / ΔΔ
- Save MFCC to features/mfccs/<stem>.npy
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from utils import config
from week1_io.io_and_plots import load_wav
from week2_stft.stft import stft_magnitude
from week3_filterbank.filterbanks_mfcc import (
    mel_triangular_basis, mfcc, hz_to_mel, mel_to_hz
)

# ------- Ordered search for wav: prioritize week1_####.wav, choose the latest -------
def _sorted_wavs(audio_dir: Path) -> list[Path]:
    wavs = list(audio_dir.rglob("*.wav"))
    if not wavs:
        return []
    pat = re.compile(r"^week1_(\d{1,})\.wav$", re.IGNORECASE)

    def keyfunc(p: Path):
        m = pat.match(p.name)
        if m:
            try:
                return (0, int(m.group(1)))
            except ValueError:
                pass
        return (1, p.name.lower())
    return sorted(wavs, key=keyfunc)

def find_default_wav() -> str | None:
    """Select a wav under data/audio (largest index preferred)."""
    audio_dir = getattr(config, "AUDIO_DIR", (config.PROJECT_ROOT / "data" / "audio"))
    audio_dir = Path(audio_dir)
    if audio_dir.exists():
        wavs = _sorted_wavs(audio_dir)
        if wavs:
            return str(wavs[-1])
    return None

def plot_mel_basis(
    B: np.ndarray,
    centers_hz: np.ndarray,
    fs: int,
    nfft: int,
    save_path: str | None = None,
    dpi: int | None = None
):
    """Visualize the Mel triangular filterbank (frequency axis in Hz)."""
    nbins = nfft // 2
    freqs = np.linspace(0, fs/2, nbins, endpoint=False)
    plt.figure(figsize=(8, 4))
    for i in range(B.shape[0]):
        plt.plot(freqs, B[i], linewidth=0.8)
    plt.title(f"Mel triangular filterbank (n_mels={B.shape[0]})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=(dpi or config.PLOT_DPI), bbox_inches="tight")
        print(f"[Week3 Demo] Mel-basis figure saved to: {save_path}")
    plt.show()

def plot_logmel(
    mag_half: np.ndarray,
    B: np.ndarray,
    fs: int,
    title: str = "Log-Mel Spectrogram",
    save_path: str | None = None,
    dpi: int | None = None
):
    """Map the STFT half spectrum to Mel and plot a log-mel heatmap (dB)."""
    eps = getattr(config, "EPS", 1e-12)
    mel = np.maximum(mag_half @ B.T, eps)
    Mdb = 20.0 * np.log10(mel)
    T, M = Mdb.shape
    extent = [0, (T - 1) * config.FRAME_SHIFT / fs, 0, M]  # y: Mel channel index
    plt.figure(figsize=(8, 4))
    plt.imshow(Mdb.T, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Time (s)"); plt.ylabel("Mel channel")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=(dpi or config.PLOT_DPI), bbox_inches="tight")
        print(f"[Week3 Demo] Log-mel figure saved to: {save_path}")
    plt.show()

def plot_feats(
    feats: np.ndarray,
    title: str,
    save_path: str | None = None,
    dpi: int | None = None
):
    """Generic heatmap plotting: input [T, D]."""
    plt.figure(figsize=(8, 4))
    plt.imshow(feats.T, origin="lower", aspect="auto")
    plt.xlabel("Frame"); plt.ylabel("Feature dim")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=(dpi or config.PLOT_DPI), bbox_inches="tight")
        print(f"[Week3 Demo] Feature figure saved to: {save_path}")
    plt.show()

def main():
    # Read wav (exit if not found)
    if len(sys.argv) >= 2:
        wav_path = sys.argv[1]
        x, fs = load_wav(wav_path)
        stem = Path(wav_path).stem
        print(f"[Week3 Demo] Using WAV: {wav_path} (fs={fs})")
    else:
        wav_path = find_default_wav()
        if wav_path:
            x, fs = load_wav(wav_path)
            stem = Path(wav_path).stem
            print(f"[Week3 Demo] Using WAV: {wav_path} (fs={fs})")
        else:
            print("[Week3 Demo] No wav found under:", config.AUDIO_DIR)
            print("Hint: record one in Week1 or run:")
            print("  python scripts/02_week3_demo.py path/to/a.wav")
            sys.exit(1)

    # pic category directories (same as Week1/2: organized by category)
    melbasis_dir   = getattr(config, "PIC_MELBASIS_DIR",   config.PROJECT_ROOT / "data" / "pic" / "melbasis")
    logmel_dir     = getattr(config, "PIC_LOGMEL_DIR",     config.PROJECT_ROOT / "data" / "pic" / "logmel")
    mfcc_dir       = getattr(config, "PIC_MFCC_DIR",       config.PROJECT_ROOT / "data" / "pic" / "mfcc")
    delta_dir      = getattr(config, "PIC_DELTA_DIR",      config.PROJECT_ROOT / "data" / "pic" / "delta")
    deltadelta_dir = getattr(config, "PIC_DELTADELTA_DIR", config.PROJECT_ROOT / "data" / "pic" / "deltadelta")
    for d in [melbasis_dir, logmel_dir, mfcc_dir, delta_dir, deltadelta_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1) Visualize Mel triangular filterbank
    B, centers_hz = mel_triangular_basis(
        fs=fs, nfft=config.NFFT, n_mels=config.NUM_MEL_FILTERS,
        fmin=config.FMIN, fmax=config.FMAX
    )
    plot_mel_basis(
        B, centers_hz, fs=fs, nfft=config.NFFT,
        save_path=str(Path(melbasis_dir) / f"{stem}.png"),
        dpi=config.PLOT_DPI
    )

    # 2) Plot Log-Mel spectrogram
    mag_half = stft_magnitude(x, nfft=config.NFFT)  # [T, nfft//2]
    if mag_half.size == 0:
        print("[Week3 Demo] STFT is empty; audio too short vs frame config.")
        sys.exit(1)
    plot_logmel(
        mag_half, B, fs=fs,
        title="Log-Mel Spectrogram (from STFT half)",
        save_path=str(Path(logmel_dir) / f"{stem}.png"),
        dpi=config.PLOT_DPI
    )

    # 3) Compute MFCC/Δ/ΔΔ and visualize
    feats = mfcc(x, fs=fs)  # [T, D]
    T, D = feats.shape
    print(f"[Week3 Demo] MFCC shape = [T={T}, D={D}]")

    # Split static/Δ/ΔΔ (per config)
    D0 = config.NUM_CEPS
    start = 0
    static = feats[:, start:start + D0]; start += D0
    plot_feats(
        static, title="MFCC (static)",
        save_path=str(Path(mfcc_dir) / f"{stem}.png"),
        dpi=config.PLOT_DPI
    )

    if config.USE_DELTA:
        delta = feats[:, start:start + D0]; start += D0
        plot_feats(
            delta, title="Delta",
            save_path=str(Path(delta_dir) / f"{stem}.png"),
            dpi=config.PLOT_DPI
        )

    if config.USE_DELTA_DELTA:
        deltadelta = feats[:, start:start + D0]
        plot_feats(
            deltadelta, title="Delta-Delta",
            save_path=str(Path(deltadelta_dir) / f"{stem}.png"),
            dpi=config.PLOT_DPI
        )

    # 4) Save to features/mfccs/<stem>.npy (avoid overwrite)
    out_dir = getattr(config, "MFCC_DIR", (config.PROJECT_ROOT / "features" / "mfccs"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"{stem}.npy"
    np.save(out, feats.astype(np.float32))
    print(f"[Week3 Demo] Saved MFCC to: {out}")

if __name__ == "__main__":
    main()
