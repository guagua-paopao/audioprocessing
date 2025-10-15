"""
Week 1 (Lab 1): recording, playback, save/load, time-domain and spectrogram display
Lab 1 Sound analysis
"""
from __future__ import annotations
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from utils import config

def record(seconds: float = 5.0, fs: int = None) -> np.ndarray:
    """Record with sounddevice (mono)."""
    fs = fs or config.FS
    print(f"[Week1] Recording {seconds}s @ {fs} Hz ...")
    x = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return x.squeeze()

def play(x: np.ndarray, fs: int = None) -> None:
    """Play audio."""
    fs = fs or config.FS
    sd.play(x, fs)
    sd.wait()

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize amplitude to [-1, 1], see Lab 1."""
    m = np.max(np.abs(x)) + 1e-12
    return 0.99 * x / m

def save_wav(path: str, x: np.ndarray, fs: int = None) -> None:
    """Save as WAV (float32)."""
    fs = fs or config.FS
    sf.write(path, x.astype(np.float32), fs)

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read WAV, return (signal, sample rate)."""
    x, fs = sf.read(path, dtype="float32")
    return x.squeeze(), fs

def plot_waveform(
    x: np.ndarray,
    fs: int = None,
    title: str = "Waveform",
    save_path: str | None = None,
    dpi: int = 300
) -> None:
    """Time-domain waveform (x-axis in seconds), see Lab 1. Supports saving image to save_path."""
    fs = fs or config.FS
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Week1] Waveform figure saved to: {save_path}")
    plt.show()

def plot_spectrogram(
    x: np.ndarray,
    fs: int = None,
    nfft: int = 512,
    win_len: int = 512,
    noverlap: int = 400,
    wideband: bool = False,
    title: str = "Spectrogram",
    save_path: str | None = None,
    dpi: int = 300
):
    """Spectrogram, narrowband/wideband (Lab 1 example). Supports saving image to save_path."""
    fs = fs or config.FS
    if wideband:
        win_len = 64
        noverlap = 40
        nfft = 64
        pad_to = 512
        plt.figure()
        plt.specgram(x, window=np.hamming(win_len), noverlap=noverlap, NFFT=nfft, pad_to=pad_to, Fs=fs)
    else:
        plt.figure()
        plt.specgram(x, window=np.hamming(win_len), noverlap=noverlap, NFFT=nfft, Fs=fs)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title + (" (Wideband)" if wideband else " (Narrowband)"))
    cbar = plt.colorbar(label="Power/Frequency (dB/Hz)")
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Week1] Spectrogram figure saved to: {save_path}")
    plt.show()
