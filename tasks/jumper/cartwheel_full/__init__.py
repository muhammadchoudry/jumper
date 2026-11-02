"""jumper.cartwheel_full -- the whole cartwheel in one policy.

Reference 4 (`../cartwheel_jump/ref/cartwheel_jump.npz`): recorded trot, sideways
jump, leaning landing on one leg planted ahead, roll on edge, tip back down, stand.
Rewards and terminations change with the reference's phase:

  A  stand, run-up, jump, landing -- track the reference (the jump task's terms)
  B  roll                         -- roll freely on edge (jumper.cartwheel_roll's terms)
  C  tip down, stand              -- get back on its feet and settle

Warm-started from the jumper.cartwheel_roll policy, which already rolls 7-9 turns
in 5 s (docs/jumper_cartwheel.md).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_full",
    assets=JUMPER_ASSETS,
    description="jumper hexapod full cartwheel: trot, sideways jump, roll on edge with free legs, tip down and stand (cartwheel_jump.npz, phase-gated rewards)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
