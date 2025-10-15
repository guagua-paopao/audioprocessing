"""
Week 2 (Lab 2): Short-time spectral processing — magAndPhase, frame-by-frame processing
Lab 2 Short-time spectral processing
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from utils import config

def magAndPhase(speech_frame: np.ndarray, nfft: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute short-time magnitude and phase spectra (Lab 2: Hamming -> FFT -> abs/angle).
    Input shape: (N,), output magnitude and phase shape: (nfft,)
    """
    nfft = nfft or config.NFFT
    frame = speech_frame.astype(np.float32)
    frame = frame * np.hamming(len(frame))
    X = np.fft.fft(frame, n=nfft)
    mag = np.abs(X)
    phase = np.angle(X)
    return mag, phase

def frame_signal(x: np.ndarray, frame_length: int = None, frame_shift: int = None) -> np.ndarray:
    """Framing (no padding): returns [num_frames, frame_length]."""
    L = frame_length or config.FRAME_LENGTH
    S = frame_shift or config.FRAME_SHIFT
    if len(x) < L:
        return np.empty((0, L), dtype=np.float32)
    num = 1 + (len(x) - L) // S
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(num, L), strides=(x.strides[0]*S, x.strides[0])
    ).copy()
    return frames

def stft_magnitude(x: np.ndarray, nfft: int = None) -> np.ndarray:
    """Magnitude spectra of the whole utterance (positive-frequency half only). Returns [num_frames, nfft//2]."""
    nfft = nfft or config.NFFT
    frames = frame_signal(x, config.FRAME_LENGTH, config.FRAME_SHIFT)
    mags = []
    for f in frames:
        mag, _ = magAndPhase(f, nfft=nfft)
        mags.append(mag[: nfft//2])
    return np.stack(mags, axis=0) if len(mags) else np.empty((0, nfft//2), dtype=np.float32)

def plot_mag_phase(
    mag: np.ndarray,
    phase: np.ndarray,
    fs: int = None,
    nfft: int = None,
    title: str = "Mag/Phase",
    save_path: str | None = None,
    dpi: int = 300
):
    """Plot magnitude and phase spectra (as suggested in Lab 2). Supports saving image to save_path."""
    fs = fs or config.FS
    nfft = nfft or config.NFFT
    faxis = np.arange(nfft) * fs / nfft
    plt.figure(figsize=(8,5))
    plt.subplot(2,1,1)
    plt.plot(faxis[:nfft//2], mag[:nfft//2])
    plt.title(title)
    plt.ylabel("Magnitude")
    plt.subplot(2,1,2)
    plt.plot(faxis[:nfft//2], phase[:nfft//2])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Week2] Mag/Phase figure saved to: {save_path}")
    plt.show()
