# noise_mix.py
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
CLEAN_ROOT = Path("data/audio")  # 干净语音 data/audio/<label>/*.wav
NOISE_DIR = Path("data/noises")  # 噪声库：若是长噪声，放几个 .wav
OUT_ROOT = Path("data_noisy")  # 输出根目录：data_noisy/20dB/<label>/*.wav 等
SNR_LIST = [20, 10, 0]


def mix_snr(x, d, snr_db):
    # Lab5：SNR 公式与功率估计（均值平方）  :contentReference[oaicite:22]{index=22}
    Px = np.mean(x ** 2)
    Pd = np.mean(d ** 2)
    target_ratio = 10 ** (-snr_db / 10)  # Pd_scaled / Px
    a = np.sqrt(target_ratio * Px / (Pd + 1e-12))
    y = x + a * d
    # 归一化避免削顶
    y = 0.99 * y / max(1e-9, np.max(np.abs(y)))
    return y


# 批处理你的 clean 语音目录，生成 20/10/0 dB 三种


def read_wav(p):
    x, fs = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    assert fs == SR, f"Sample rate mismatch: {fs} != {SR}"
    # 轻微归一化（可选）
    x = x / max(1e-9, np.abs(x).max())
    return x


def random_noise_segment(noise_files, length):
    """从噪声库里随机挑一段长度与语音一致的片段；不足则循环拼接。"""
    import random
    n = []
    while sum(len(seg) for seg in n) < length:
        p = random.choice(noise_files)
        d = read_wav(p)
        n.append(d)
    dcat = np.concatenate(n)
    start = np.random.randint(0, len(dcat) - length + 1)
    return dcat[start:start + length]


def main():
    noise_files = sorted(NOISE_DIR.glob("*.wav"))
    assert noise_files, f"No noise wavs found in {NOISE_DIR}"

    for wav in sorted(CLEAN_ROOT.rglob("*.wav")):
        x = read_wav(wav)
        for snr in SNR_LIST:
            d = random_noise_segment(noise_files, len(x))
            y = mix_snr(x, d, snr)
            rel = wav.relative_to(CLEAN_ROOT)  # <label>/file.wav
            out_dir = OUT_ROOT / f"{snr}dB" / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / rel.name
            sf.write(out_path.as_posix(), y, SR)
            print(f"SAVED: {out_path}")


if __name__ == "__main__":
    main()
