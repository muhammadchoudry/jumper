"""jumper.cartwheel_ten_star -- the 10-turn straight cartwheel holding the star while
it rolls: every limb straight hip to tip (legs_straight) and the six tips evenly
spread, 60 deg apart round the hub in the wheel's plane (spokes_even).

The one movement left free is tilting a straight leg out of the wheel's plane,
which shortens its reach in the plane -- the only pumping a straight, evenly spread
star has. A fully rigid star cannot keep rolling (~2/3 of its spin energy is lost at
each spoke change), so this asks whether that tilt is enough.
Warm-started from jumper.cartwheel_ten_straight run 1 (straight_v1_model_15897).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_ten_star",
    assets=JUMPER_ASSETS,
    description="jumper hexapod 10-turn straight cartwheel holding the star (straight, evenly spread legs) while rolling",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
