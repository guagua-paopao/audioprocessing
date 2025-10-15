
"""
Global configuration
Parameters designed per Lab 1–5 for centralized modification.
"""

from pathlib import Path

# =========================
# Sampling and framing
# =========================

# Sample rate (Lab 1 suggests 16 kHz)
FS = 16000

# Frame length and frame shift (Lab 2: 20 ms frames, 50% overlap)
FRAME_LENGTH = int(0.020 * FS)  # 320 samples
FRAME_SHIFT  = FRAME_LENGTH // 2

# FFT points (Lab 3 uses 512-point example)
NFFT = 512

# Filterbanks (Lab 3)
NUM_MEL_FILTERS = 40
FMIN = 0.0
FMAX = FS / 2.0

# MFCC (number of coefficients after DCT; whether to include energy can be decided during extraction)
NUM_CEPS = 13
INCLUDE_LOG_ENERGY = True
USE_DELTA = True
USE_DELTA_DELTA = True

# =========================
# Image directories
# =========================
# Data and feature directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
MFCC_DIR  = PROJECT_ROOT / "features" / "mfccs"
PIC_DIR = PROJECT_ROOT / "data" / "pic"

# Week1: organized by category
PIC_WAVEFORM_DIR    = PIC_DIR / "waveform"
PIC_SPECTROGRAM_DIR = PIC_DIR / "spectrogram"

# Week2: organized by category
PIC_MAGPHASE_DIR    = PIC_DIR / "magphase"
PIC_STFT_DIR        = PIC_DIR / "stft"

# Week3: organized by category
PIC_MELBASIS_DIR    = PIC_DIR / "melbasis"
PIC_LOGMEL_DIR      = PIC_DIR / "logmel"
PIC_MFCC_DIR        = PIC_DIR / "mfcc"
PIC_DELTA_DIR       = PIC_DIR / "delta"
PIC_DELTADELTA_DIR  = PIC_DIR / "deltadelta"


# =========================
# Training-related
# =========================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-2

# Random seed
RANDOM_STATE = 0

# Numerical stability constant and default DPI for plotting
EPS = 1e-12          # e.g., for log/normalization scenarios
PLOT_DPI = 300       # default DPI for all saved images

# Prefix and width for auto-numbered recordings (aligned with Week1 auto-naming)
AUDIO_PREFIX = "week1_"
AUDIO_INDEX_WIDTH = 4  # width like 0001

