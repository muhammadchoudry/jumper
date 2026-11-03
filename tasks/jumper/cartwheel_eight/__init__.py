"""jumper.cartwheel_eight -- the full cartwheel, rolling a figure 8.

`jumper.cartwheel_full`'s phase-gated task (cartwheel_full/env_cfg.py `build`) on
reference 5 (`ref/cartwheel_eight.npz`, tools/crab_trio/cartwheel_ref5.py): trot, sideways
jump, a roll segment ~5.5 s long (about 10 turns at the policy's pace), tip down,
stand. The roll is scored against the reference's heading at each moment, its lean against what the turn needs (tan(lean) = v * yaw rate / g, as a rolling coin leans), so it learns to steer.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_eight",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel rolling a figure 8: trot, sideways jump, roll a loop left then a loop right leaning into the turns, tip down and stand (cartwheel_eight.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
