"""Reference 5: reference 4 with a long roll -- 10 turns -- straight, or as a figure 8.

Trot, sideways jump onto one leg, leaning landing, then the roll: TURNS turns of the
rigid star at ROLL_RATE (the policy rolls at its own pace; the roll segment's length
is what matters, ~5.5 s, about what 10 turns take it), then tip down and stand.

  straight   heading fixed
  eight      EIGHT_TURNS turns (a longer roll): two round loops, one left and one
             right, joined by straight diagonals that cross in the middle -- constant
             lean around each loop, upright on the diagonals, so the lean changes
             sides with time to stand up in between. (The first version was two
             tangent circles: the lean had to flip from left to right in ~0.4 s, and
             the trained policy rolled the left loop and fell at the switch.)

  python cartwheel_ref5.py   # writes tasks/jumper/cartwheel_ten/ref/cartwheel_ten.npz,
                             #    tasks/jumper/cartwheel_eight/ref/cartwheel_eight.npz
                             #    and .../cartwheel_long.npz (straight, the 8's roll length)
"""
from __future__ import annotations

import math

import cartwheel_ref4 as R4
from cartwheel_ref import mj

TURNS = 4.0          # reference turns in the roll (rigid star at ROLL_RATE)
ROLL_RATE = 4.7      # rad/s: ~1.4 m/s at the rim, the policy's own rolling speed; ~5.5 s of roll


def eight(s: float) -> float:
    """figure 8: left loop then right loop"""
    blend = 0.08                                     # ease the yaw rate through the crossover
    if s < 0.5 - blend:
        return 2 * math.pi * s / 0.5
    if s > 0.5 + blend:
        return 2 * math.pi * (1.0 - s) / 0.5
    a = (s - (0.5 - blend)) / (2 * blend)            # 0..1 across the crossover
    up = 2 * math.pi * s / 0.5
    down = 2 * math.pi * (1.0 - s) / 0.5
    return up + mj(a) * (down - up)


EIGHT_TURNS = 5.5     # a longer roll for the figure 8: bigger loops, less lean
BETA = math.radians(30)   # the crossing diagonals are 2*BETA apart


def _eight_profile(n: int = 4000):
    """Heading vs fraction of the path for a figure 8 of two round loops joined by
    straight diagonals that cross in the middle: half a diagonal, loop left by
    pi + 2 BETA, the other diagonal through the crossing, loop right by pi + 2 BETA,
    half a diagonal back to the crossing. Curvature is constant on the loops (constant
    lean) and zero on the diagonals (stand up before leaning the other way), eased
    over a short stretch at each change."""
    import numpy as np
    R = 1.0
    d = R / math.sin(BETA)
    half = math.sqrt(d * d - R * R)                  # crossing to a tangent point
    arc = R * (math.pi + 2 * BETA)
    parts = [(half, 0), (arc, 1), (2 * half, 0), (arc, -1), (half, 0)]
    total = sum(p[0] for p in parts)
    s = (np.arange(n) + 0.5) / n
    edges = np.cumsum([0] + [p[0] / total for p in parts])
    rate = np.zeros(n)
    e = 0.02
    for (length, sign), a, b in zip(parts, edges[:-1], edges[1:]):
        if sign:
            ramp = np.minimum(np.clip((s - a) / e, 0, 1), np.clip((b - s) / e, 0, 1))
            ramp = ramp * ramp * ramp * (10 - 15 * ramp + 6 * ramp * ramp)
            seg = np.where((s >= a) & (s < b), ramp, 0.0)
            seg *= (math.pi + 2 * BETA) / (seg.sum() / n)
            rate += sign * seg
    psi = np.concatenate([[0.0], np.cumsum(rate) / n])
    return np.linspace(0, 1, n + 1), psi


_EIGHT = None


def eight3(s: float) -> float:
    """figure 8: two round loops, straight diagonals crossing in the middle"""
    import numpy as np
    global _EIGHT
    if _EIGHT is None:
        _EIGHT = _eight_profile()
    return float(np.interp(s, *_EIGHT))


def main():
    R4.TURNS, R4.ROLL_RATE = TURNS, ROLL_RATE
    R4.HEADING_FN = None
    R4.OUT = R4.ROOT / "tasks/jumper/cartwheel_ten/ref/cartwheel_ten.npz"
    R4.build()
    R4.TURNS = EIGHT_TURNS                       # straight, the 8's roll length: spawn states for
    R4.HEADING_FN = None                         # cartwheel_eight try 3 (it draws its 8 by distance)
    R4.OUT = R4.ROOT / "tasks/jumper/cartwheel_eight/ref/cartwheel_long.npz"
    R4.build()
    R4.HEADING_FN = eight3
    R4.OUT = R4.ROOT / "tasks/jumper/cartwheel_eight/ref/cartwheel_eight.npz"
    R4.build()


if __name__ == "__main__":
    main()
