"""jumper.cartwheel_ten -- the full cartwheel, rolling ten turns in a straight line.

`jumper.cartwheel_full`'s phase-gated task (cartwheel_full/env_cfg.py `build`) on
reference 5 (`ref/cartwheel_ten.npz`, tools/crab_trio/cartwheel_ref5.py): trot, sideways
jump, a roll segment ~5.5 s long (about 10 turns at the policy's pace), tip down,
stand.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_ten",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel with a long roll: trot, sideways jump, ~10 turns on edge in a straight line, tip down and stand (cartwheel_ten.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
