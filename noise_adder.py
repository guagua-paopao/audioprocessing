import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

speech, fs_s = sf.read("data/audio/amber010.wav")
noise, fs_n = sf.read("noise.wav")
assert fs_s == fs_n
fs = fs_s
speech = speech[:45000]


def power(signal):
    return np.mean(signal**2)


def snr(s, n):
    return 10 * np.log10(power(s) / power(n))


def noise_scaling(s, n, r):
    return np.sqrt((power(s) / power(n)) * np.pow(10, -r / 10))


# sd.play(speech, fs)
# sd.wait()
# sd.play(noise, fs)
# sd.wait()

print(snr(speech, noise))
mix1 = speech + noise
# sd.play(mix1,fs)
# sd.wait()
plt.plot(speech)
plt.show()
plt.plot(noise)
plt.show()
plt.plot(mix1)
plt.show()
# sd.play(mix1, fs)
# sd.wait()
print(snr(mix1, noise))
alpha = noise_scaling(speech, noise, 10)
print(alpha)
mix2 = (alpha * noise) + speech
# print(noise_scaling(speech, noise, 10))
print(snr(mix2, noise * alpha))
plt.plot(mix2)
plt.show()
sd.play(mix2, fs)
sd.wait()
