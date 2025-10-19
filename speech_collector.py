import os
from random import randint
import sounddevice as sd
import soundfile as sf

name_file = open("NAMES.txt")
names = [name.strip() for name in name_file.readlines()]
name_file.close()
names_dict = dict()
for n in names:
    names_dict[n] = 0

TARGET = 20

#
for filename in os.listdir("recordings/"):
    if filename[:-7].title() in names and filename[-4:] == ".wav":
        name = filename[:-7].title()
        names_dict[name] += 1

for k, v in names_dict.items():
    print(f"{v:>3} {k}")
    if v >= TARGET:
        names.remove(k)




FS = 16000
SECONDS = 3
running = (len(names) > 0)

while running:
    name = names[randint(0, len(names) - 1)]
    while names_dict[name] >= TARGET and len(names) > 0:
        names_dict.pop(name)
        names.remove(name)
        name = names[randint(0, len(names) - 1)]
    if len(names) == 0:
        break
    cmd = input(f"{name} ({names_dict[name]}/{TARGET})").strip()
    if cmd == "" or cmd == "p":
        # record
        r = sd.rec(SECONDS * FS, samplerate=FS, channels=1)
        print(f"RECORDING...")
        sd.wait()
        r_norm = 0.99 * r / max(abs(r))

        # save recording "name123.wav"
        out_name = f"recordings/{name.lower()}{names_dict[name] + 1:03}.wav"
        sf.write(out_name, r_norm, FS)
        if cmd == "p":
            print("PLAYING...")
            sd.play(r_norm, FS)
        names_dict[name] += 1
    else:
        running = False
print("Stopping...")
