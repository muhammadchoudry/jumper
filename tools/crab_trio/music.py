"""Original synth track for Reef Rave, written beat-for-beat against choreo.py.

120 BPM, A minor, 34.5 s. Accents land on the routine's events:
  beats 6/7/8 and 10/11/12   rising then falling 'blips' as each crab raises / lowers its claws
  beats 15-20                a clap on every pincer snap
  beats 22-28                filter swells with the body twists
  beats 31-38                four-on-the-floor with an offbeat bass for the claw pumps
  beats 40/41/42             blips again for the right-to-left ripple
  beats 44-58                half-time build; a 'boing' at each takeoff and a thump on each landing
  beats 60/61/62             big hits on the three unison bounces
  beat 63.5                  the bow: final chord and a downward sweep

  python tools/crab_trio/music.py out.wav
"""
from __future__ import annotations

import sys
import wave

import numpy as np
from scipy.signal import butter, lfilter

SR = 44100
BPM = 120
BEAT = 60 / BPM
LENGTH = 34.5
N = int(SR * LENGTH)
rng = np.random.default_rng(7)


def at(beat: float) -> int:
    return int(beat * BEAT * SR)


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def env(n: int, attack=0.005, decay=0.2, sustain=0.0, release=0.05) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(attack, 1e-4))
    e *= sustain + (1 - sustain) * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))
    rel = int(release * SR)
    if rel and n > rel:
        e[-rel:] *= np.linspace(1, 0, rel)
    return e


def lowpass(x, cutoff):
    b, a = butter(2, min(cutoff, SR / 2 - 100) / (SR / 2))
    return lfilter(b, a, x)


def highpass(x, cutoff):
    b, a = butter(2, cutoff / (SR / 2), "high")
    return lfilter(b, a, x)


def saw(f, n, detune=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-detune, 0, detune):
        out += 2 * ((t * f * (1 + d)) % 1) - 1
    return out / 3


L = np.zeros(N)
R = np.zeros(N)


def put(sig, beat, pan=0.0, gain=1.0):
    i = at(beat)
    sig = sig[: max(0, N - i)]
    L[i:i + len(sig)] += sig * gain * np.sqrt((1 - pan) / 2)
    R[i:i + len(sig)] += sig * gain * np.sqrt((1 + pan) / 2)


# ---- instruments ------------------------------------------------------------
def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.12)


def clap():
    n = int(0.25 * SR)
    noise = highpass(rng.standard_normal(n), 900)
    e = env(n, 0.001, 0.06)
    for d in (0.01, 0.02):  # the three-hand flam that makes a clap
        k = int(d * SR)
        e[k:] += 0.6 * env(n - k, 0.001, 0.02)
    return noise * e * 0.5


def snap():  # pincer click: bright and short
    n = int(0.06 * SR)
    return highpass(rng.standard_normal(n), 4000) * env(n, 0.0005, 0.012) * 0.6


def hat(open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    return highpass(rng.standard_normal(n), 7000) * env(n, 0.001, 0.08 if open_ else 0.015) * 0.25


def bass(note, beats):
    n = int(beats * BEAT * SR)
    x = saw(hz(note), n, 0.003) + 0.5 * np.sin(2 * np.pi * hz(note - 12) * np.arange(n) / SR)
    return lowpass(x, 500) * env(n, 0.005, 0.25, 0.4, 0.03) * 0.5


def pad(notes, beats, cutoff=1200):
    n = int(beats * BEAT * SR)
    x = sum(saw(hz(m), n, 0.006) for m in notes) / len(notes)
    return lowpass(x, cutoff) * env(n, 0.25, 1.0, 0.8, 0.3) * 0.35


def blip(note):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * hz(note) * t)) * 0.4 + np.sin(2 * np.pi * hz(note + 12) * t)
    return lowpass(x, 3500) * env(n, 0.002, 0.09) * 0.35


def pluck(note):
    n = int(0.25 * SR)
    return lowpass(saw(hz(note), n), 2600) * env(n, 0.002, 0.07) * 0.22


def riser(beats, up=True):
    n = int(beats * BEAT * SR)
    t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    for k, c in enumerate(np.linspace(400, 9000, 24) if up else np.linspace(9000, 300, 24)):
        s = slice(k * n // 24, (k + 1) * n // 24)
        out[s] = highpass(noise[s], c)
    return out * (t if up else 1 - t) * 0.18


def impact():
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 80 * np.exp(-t / 0.05)) / SR) * env(n, 0.001, 0.4)
    return boom + lowpass(rng.standard_normal(n), 1500) * env(n, 0.001, 0.2) * 0.3


# ---- the arrangement -------------------------------------------------------
# A minor: Am F C G, one chord per bar (4 beats)
CHORDS = [[57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62]]
ROOTS = [45, 41, 48, 43]
ARP = [0, 1, 2, 1]

# intro (0-6): pad in, riser into the first claw raise
put(pad(CHORDS[0], 6, 700), 0, gain=0.8)
put(riser(4), 2)

for bar in range(1, 13):
    b0 = bar * 4 + 2  # bars start on even beats from beat 6
    if b0 >= 44:
        break
    ch, root = CHORDS[(bar - 1) % 4], ROOTS[(bar - 1) % 4]
    sweep = 900 + 1800 * (0.5 + 0.5 * np.sin(bar)) if 22 <= b0 < 31 else 1500
    put(pad(ch, 4, sweep), b0, gain=0.6)
    for i in range(4):
        b = b0 + i
        if b >= 44:
            break
        put(kick(), b, gain=0.9 if b >= 14 else 0.55)
        put(hat(), b + 0.5, pan=0.3)
        if b >= 21:
            put(hat(), b + 0.25, pan=-0.3, gain=0.6)
            put(hat(), b + 0.75, pan=-0.3, gain=0.6)
        if 31 <= b < 39:  # pumps: offbeat bass and open hats
            put(bass(root, 0.5), b + 0.5)
            put(hat(True), b + 0.5, pan=0.2)
        else:
            put(bass(root, 1.0), b)
        if b in (17, 19, 25, 27, 33, 35, 37):
            put(clap(), b, gain=0.8)
        if b >= 21 and b < 44:
            for s in range(2):
                put(pluck(ch[ARP[(i * 2 + s) % 4]] + 12), b + 0.5 * s, pan=0.4 * (-1) ** s)

# ripple up / down, left to right (crab 0 is left: pan accordingly)
for k in range(3):
    put(blip(69 + 3 * k), 6 + k, pan=-0.6 + 0.6 * k)
    put(blip(76 - 3 * k), 10 + k, pan=-0.6 + 0.6 * k)
# pincer snaps on every beat 15-20, clap layered under them
for b in range(15, 21):
    put(snap(), b + 0.06, pan=0.0)
    put(clap(), b, gain=0.5)
put(riser(2), 19)
# reverse ripple, right to left
for k in range(3):
    put(blip(72 + 4 * k), 40 + (2 - k), pan=0.6 - 0.6 * (2 - k))
put(riser(2), 42)
# jumps (44-58): the groove drops to a tense half-time pulse; each jump gets a
# rising 'boing' at takeoff and a thump on landing, panned to the crab that jumps.
TAKEOFF, LAND = -0.28, 0.55   # beats relative to the apex, from the jump data


def boing(note):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = hz(note) * (1 + 1.5 * (1 - np.exp(-t / 0.08)))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.003, 0.18) * 0.45


def thump():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * np.cumsum(60 + 60 * np.exp(-t / 0.02)) / SR) * env(n, 0.001, 0.08)


put(pad([45, 52, 57, 60], 14, 600), 44, gain=0.7)
for b in range(44, 58, 2):
    put(kick(), b, gain=0.6)
    put(hat(), b + 1, gain=0.6)
for k, apex in enumerate((48, 50, 52)):
    pan = -0.6 + 0.6 * k
    put(riser(1.2), apex - 1.5, pan=pan)
    put(boing(60 + 4 * k), apex + TAKEOFF, pan=pan)
    put(thump(), apex + LAND, pan=pan, gain=0.8)
    put(clap(), apex + LAND, pan=pan, gain=0.6)
put(riser(3.5), 52.2)                       # big build into the unison jump
put(boing(72), 56 + TAKEOFF)
put(boing(67), 56 + TAKEOFF, pan=-0.6)
put(boing(76), 56 + TAKEOFF, pan=0.6)
put(impact(), 56 + LAND, gain=0.9)
put(clap(), 56 + LAND, gain=0.9)

# groove back in for the claws-high finale
for b in range(57, 63):
    put(kick(), b + 0.5 if b == 57 else b, gain=0.9)
    put(hat(), b + 0.5, pan=0.3)
    put(bass(45 if b < 60 else 43, 1.0), b)
put(pad(CHORDS[0], 3, 1400), 57, gain=0.6)
# finale hits on the three bounces
for i, b in enumerate((60, 61, 62)):
    put(impact(), b, gain=0.55 + 0.1 * i)
    put(clap(), b, gain=0.9)
    put(pad([57 + 12 * (i == 2), 60, 64, 69], 1, 3000), b, gain=0.7)
# the bow: final chord rings out with a downward sweep
put(riser(2.5, up=False), 63.5)
put(pad([45, 57, 60, 64, 69], 5, 1600), 63.5, gain=1.1)
put(impact(), 63.5, gain=0.8)

# ---- master ----------------------------------------------------------------
mix = np.stack([R, L], axis=1)  # crab 0 stands on the right of frame
mix = np.tanh(mix * 1.6) / np.tanh(1.6)       # gentle saturation / glue
mix *= 0.89 / np.max(np.abs(mix))
fade = int(1.0 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]

out = sys.argv[1] if len(sys.argv) > 1 else "reef_rave.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out)
