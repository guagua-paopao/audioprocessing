"""
Week 3 (Lab 3): Filterbanks and MFCC
- Linear rectangular filterbank (summation + basis/matmul)
- Mel triangular filterbank
- MFCC (optional Δ/ΔΔ)
Reference: Lab 3 Filterbanks
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.fftpack import dct
from utils import config
from week2_stft.stft import stft_magnitude

# -------- Basic utilities --------
def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

# -------- Linear rectangular filterbank (summation) --------
def linear_rect_fbank_sum(mag_spec_half: np.ndarray, num_ch: int = 8) -> np.ndarray:
    """
    Input: single-frame half spectrum (nfft/2,)
    Output: filterbank energies (num_ch,)
    """
    N = mag_spec_half.shape[0]
    assert N % num_ch == 0, "为了简单起见，N 应能被 num_ch 整除"
    band = N // num_ch
    fbank = np.zeros((num_ch,), dtype=np.float32)
    for i in range(num_ch):
        fbank[i] = np.sum(mag_spec_half[i*band:(i+1)*band])
    return fbank

# -------- Linear rectangular filterbank (basis/matmul) --------
def linear_rect_basis(nbins: int, num_ch: int = 8) -> np.ndarray:
    """Generate basis matrix for linear rectangular filterbank, shape=[num_ch, nbins]"""
    assert nbins % num_ch == 0
    band = nbins // num_ch
    B = np.zeros((num_ch, nbins), dtype=np.float32)
    for i in range(num_ch):
        B[i, i*band:(i+1)*band] = 1.0
    return B

# -------- Mel triangular filterbank (matrix implementation) --------
def mel_triangular_basis(
    fs: int = None,
    nfft: int = None,
    n_mels: int = None,
    fmin: float = None,
    fmax: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Mel triangular filterbank basis matrix (one triangle per row).
    Returns (B, f_centers_hz)
    B shape = [n_mels, nfft//2]
    """
    fs   = fs or config.FS
    nfft = nfft or config.NFFT
    n_mels = n_mels or config.NUM_MEL_FILTERS
    fmin = fmin if fmin is not None else config.FMIN
    fmax = fmax if fmax is not None else config.FMAX

    nbins = nfft // 2
    # Frequency to Mel
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)  # includes boundary extension points
    hz_points = mel_to_hz(mel_points)

    # Map to spectrum bins
    bin_freqs = np.linspace(0, fs/2, nbins, endpoint=False)  # half-spectrum frequencies
    def hz_to_bin(hz):
        # Find the nearest bin index
        return int(np.floor(np.searchsorted(bin_freqs, hz, side="left")))

    bins = np.array([hz_to_bin(h) for h in hz_points])
    B = np.zeros((n_mels, nbins), dtype=np.float32)
    for m in range(1, n_mels+1):
        left, center, right = bins[m-1], bins[m], bins[m+1]
        if right <= left:  # defensive handling
            continue
        for k in range(left, center):
            if center != left:
                B[m-1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                B[m-1, k] = (right - k) / (right - center)
    return B, hz_points[1:-1]

def apply_fbank_basis(mag_spec_half: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply filterbank basis matrix to a single-frame half spectrum -> per-channel energies."""
    return basis @ mag_spec_half

# -------- Δ / ΔΔ --------
def delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    """First-order temporal difference (along frame dimension), feat: [T, D]"""
    T, D = feat.shape
    denom = 2 * sum([i**2 for i in range(1, N+1)])
    out = np.zeros_like(feat)
    for t in range(T):
        num = np.zeros((D,), dtype=feat.dtype)
        for n in range(1, N+1):
            t_p = min(T-1, t+n)
            t_m = max(0, t-n)
            num += n * (feat[t_p] - feat[t_m])
        out[t] = num / denom
    return out

# -------- MFCC extraction --------
def mfcc(
    x: np.ndarray,
    fs: int = None,
    nfft: int = None,
    n_mels: int = None,
    num_ceps: int = None,
    include_log_energy: bool = None,
    use_delta: bool = None,
    use_delta_delta: bool = None
) -> np.ndarray:
    """
    Compute MFCC following Lab 2/3 pipeline (Mel filterbank implemented from scratch).
    Returns a [T, D] feature matrix.
    """
    eps = getattr(config, "EPS", 1e-12)

    fs   = fs or config.FS
    nfft = nfft or config.NFFT
    n_mels = n_mels or config.NUM_MEL_FILTERS
    num_ceps = num_ceps or config.NUM_CEPS
    include_log_energy = include_log_energy if include_log_energy is not None else config.INCLUDE_LOG_ENERGY
    use_delta = use_delta if use_delta is not None else config.USE_DELTA
    use_delta_delta = use_delta_delta if use_delta_delta is not None else config.USE_DELTA_DELTA

    # STFT magnitude half spectrum [T, nfft//2]
    mag = stft_magnitude(x, nfft=nfft) + eps

    # Mel filterbank
    B, _ = mel_triangular_basis(fs=fs, nfft=nfft, n_mels=n_mels, fmin=config.FMIN, fmax=config.FMAX)
    mel_energies = np.maximum(mag @ B.T, eps)  # [T, n_mels]

    # Log energies
    log_mel = np.log(mel_energies)

    # DCT -> MFCC (take first num_ceps)
    ceps = dct(log_mel, type=2, norm="ortho", axis=1)[:, :num_ceps]

    # Replace c0 with log energy (common practice)
    if include_log_energy:
        logE = np.log(np.sum(mag, axis=1) + eps)  # approximate total energy
        ceps[:, 0] = logE

    # Δ / ΔΔ
    feats = ceps
    if use_delta:
        d1 = delta(feats)
        feats = np.concatenate([feats, d1], axis=1)
    if use_delta_delta:
        d2 = delta(delta(ceps))
        feats = np.concatenate([feats, d2], axis=1)
    return feats.astype(np.float32)
