"""
Week 1 Demo:
- List audio devices (to help choose mic/speaker)
- Record 3 seconds -> normalize -> save as wav
- Plot waveform & spectrogram (and save PNG: organized by category)
- Play the just-recorded audio
"""
from pathlib import Path
import re  # Used to match already-numbered files
import numpy as np
import sounddevice as sd
from utils import config
from week1_io.io_and_plots import record, normalize, save_wav, load_wav, play, plot_waveform, plot_spectrogram

import matplotlib
matplotlib.use("TkAgg")


# def list_devices():
#     """Print available input/output devices to help selection."""
#     try:
#         devs = sd.query_devices()
#         print("\n==== Available audio devices ====")
#         for i, d in enumerate(devs):
#             print(f"[{i:2d}] {d['name']}  (in:{d['max_input_channels']}, out:{d['max_output_channels']})")
#         print("=================================\n")
#     except Exception as e:
#         print("Device enumeration failed:", e)

# --- Auto-numbered filename generator ---
def next_indexed_wav_path(out_dir: Path, prefix: str = "week1_", width: int = 4) -> Path:
    """
    In out_dir, look for files of the form prefix + number + ".wav",
    take the current maximum index + 1 to generate the next filename (zero-padded).
    Example: week1_0001.wav, week1_0002.wav, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.wav$", re.IGNORECASE)

    max_idx = 0
    for p in out_dir.glob("*.wav"):
        m = pattern.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass

    next_idx = max_idx + 1
    fname = f"{prefix}{str(next_idx).zfill(width)}.wav"
    return out_dir / fname

def main():
    # List devices; if recording fails, run once to check indices
    # list_devices()

    # Specify devices (change the following line to your device index or an (input, output) tuple)
    # sd.default.device = (input_device_index, output_device_index)
    # sd.default.device = (1, 3)

    # Set default sample rate & mono
    sd.default.samplerate = config.FS
    sd.default.channels = 1

    # Paths: audio and pic are siblings, both under data
    data_root = (config.PROJECT_ROOT / "data")
    out_dir_audio = (data_root / "audio")
    out_wav = next_indexed_wav_path(out_dir_audio, prefix="week1_", width=4)

    # Record 3 seconds
    x = record(seconds=3.0, fs=config.FS)
    x = normalize(x)
    save_wav(str(out_wav), x, fs=config.FS)
    print(f"[Demo] Saved to: {out_wav}")

    # Load the just-saved file, plot and save images (organized by category)
    y, fs = load_wav(str(out_wav))
    stem = out_wav.stem

    # pic directory and category subdirs (parallel with audio)
    pic_root = (data_root / "pic")
    wf_dir = (pic_root / "waveform")
    spec_dir = (pic_root / "spectrogram")
    wf_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    wf_png = wf_dir / f"{stem}.png"
    spec_png = spec_dir / f"{stem}.png"

    plot_waveform(y, fs=fs, title="Week1 Demo - Waveform", save_path=str(wf_png), dpi=300)
    plot_spectrogram(y, fs=fs, wideband=False, title="Week1 Demo - Spectrogram", save_path=str(spec_png), dpi=300)

    # Play
    print("[Demo] Playing...")
    play(y, fs=fs)
    print("[Demo] Done.")

if __name__ == "__main__":
    main()
