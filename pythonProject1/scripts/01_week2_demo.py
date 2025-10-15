"""
Week 2 Demo:
- Read wav (or generate test signal) -> (changed as required: exit if not found, no longer generate test tone)
- Frame the signal, take the middle frame to compute mag & phase and plot
- Compute STFT magnitude for the whole clip (half spectrum), display as a heatmap

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
from week2_stft.stft import frame_signal, magAndPhase, stft_magnitude, plot_mag_phase

# ------- Ordered search for wavs: prioritize week1_####.wav, pick the latest -------
def _sorted_wavs(audio_dir: Path) -> list[Path]:
    wavs = list(audio_dir.rglob("*.wav"))
    if not wavs:
        return []
    pat = re.compile(r"^week1_(\d{1,})\.wav$", re.IGNORECASE)

    def keyfunc(p: Path):
        m = pat.match(p.name)
        if m:
            try:
                return (0, int(m.group(1)))  # Prioritize numbered files, sort by number
            except ValueError:
                pass
        return (1, p.name.lower())          # Others sorted alphabetically
    return sorted(wavs, key=keyfunc)

def find_default_wav() -> str | None:
    audio_dir = getattr(config, "AUDIO_DIR", (config.PROJECT_ROOT / "data" / "audio"))
    audio_dir = Path(audio_dir)
    if audio_dir.exists():
        wavs = _sorted_wavs(audio_dir)
        if wavs:
            return str(wavs[-1])  # The latest one; use wavs[0] for the earliest
    return None

def plot_stft_heat(
    mag_half: np.ndarray,
    fs: int,
    nfft: int,
    title: str = "STFT magnitude (half)",
    save_path: str | None = None,
    dpi: int | None = None
):
    """Plot the [T, nfft//2] half-spectrum magnitude as a heatmap (log scale)."""
    eps = getattr(config, "EPS", 1e-12)
    Mdb = 20.0 * np.log10(np.maximum(mag_half, eps))
    T, _ = Mdb.shape
    extent = [0, (T - 1) * config.FRAME_SHIFT / fs, 0, fs / 2]  # x: seconds, y: frequency Hz
    plt.figure()
    plt.imshow(Mdb.T, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    if save_path:
        plt.gcf().savefig(save_path, dpi=(dpi or config.PLOT_DPI), bbox_inches="tight")
        print(f"[Week2 Demo] STFT figure saved to: {save_path}")
    plt.show()

def main():
    # Read wav (exit if not found)
    if len(sys.argv) >= 2:
        wav_path = sys.argv[1]
        x, fs = load_wav(wav_path)
        stem = Path(wav_path).stem
        print(f"[Week2 Demo] Using WAV: {wav_path} (fs={fs})")
    else:
        wav_path = find_default_wav()
        if wav_path:
            x, fs = load_wav(wav_path)
            stem = Path(wav_path).stem
            print(f"[Week2 Demo] Using WAV: {wav_path} (fs={fs})")
        else:
            print("[Week2 Demo] No wav found under:", config.AUDIO_DIR)
            print("Hint: record one in Week1 or run:")
            print("  python scripts/01_week2_demo.py path/to/a.wav")
            sys.exit(1)

    # pic category directories
    magphase_dir = getattr(config, "PIC_MAGPHASE_DIR", config.PROJECT_ROOT / "data" / "pic" / "magphase")
    stft_dir     = getattr(config, "PIC_STFT_DIR",     config.PROJECT_ROOT / "data" / "pic" / "stft")
    Path(magphase_dir).mkdir(parents=True, exist_ok=True)
    Path(stft_dir).mkdir(parents=True, exist_ok=True)

    # Framing
    frames = frame_signal(x, config.FRAME_LENGTH, config.FRAME_SHIFT)
    if frames.size == 0:
        # If too short, zero-pad then frame
        pad = np.pad(x, (0, max(0, config.FRAME_LENGTH - len(x))))
        frames = frame_signal(pad, config.FRAME_LENGTH, config.FRAME_SHIFT)

    # Take the middle frame for mag/phase + save plot
    idx = len(frames) // 2
    mag, phase = magAndPhase(frames[idx], nfft=config.NFFT)
    plot_mag_phase(
        mag, phase, fs=fs, nfft=config.NFFT,
        title=f"Mag/Phase (frame #{idx})",
        save_path=str(Path(magphase_dir) / f"{stem}.png"),
        dpi=config.PLOT_DPI
    )

    # Compute whole-clip STFT half-spectrum magnitude, visualize + save plot
    mag_half = stft_magnitude(x, nfft=config.NFFT)  # [T, nfft//2]
    if mag_half.size == 0:
        print("[Week2 Demo] STFT result is empty, check frame config or audio length.")
        sys.exit(1)
    plot_stft_heat(
        mag_half, fs=fs, nfft=config.NFFT,
        title="STFT magnitude (half spectrum, log)",
        save_path=str(Path(stft_dir) / f"{stem}.png"),
        dpi=config.PLOT_DPI
    )

if __name__ == "__main__":
    main()
