"""Reference 5: reference 4 with a long roll -- 10 turns -- straight, or as a figure 8.

Trot, sideways jump onto one leg, leaning landing, then the roll: TURNS turns of the
rigid star at ROLL_RATE (the policy rolls at its own pace; the roll segment's length
is what matters, ~5.5 s, about what 10 turns take it), then tip down and stand.

  straight   heading fixed
  eight      heading 0 -> 2 pi over the first half of the roll's turns (a loop to the
             left), then back 2 pi -> 0 (a loop to the right), eased through the
             crossover: a figure 8 that ends heading the way it started

  python cartwheel_ref5.py   # writes tasks/jumper/cartwheel_ten/ref/cartwheel_ten.npz
                             #    and tasks/jumper/cartwheel_eight/ref/cartwheel_eight.npz
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


def main():
    R4.TURNS, R4.ROLL_RATE = TURNS, ROLL_RATE
    R4.HEADING_FN = None
    R4.OUT = R4.ROOT / "tasks/jumper/cartwheel_ten/ref/cartwheel_ten.npz"
    R4.build()
    R4.HEADING_FN = eight
    R4.OUT = R4.ROOT / "tasks/jumper/cartwheel_eight/ref/cartwheel_eight.npz"
    R4.build()


if __name__ == "__main__":
    main()
