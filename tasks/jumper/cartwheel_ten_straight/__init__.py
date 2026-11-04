"""jumper.cartwheel_ten_straight -- the 10-turn straight cartwheel with the legs kept
straight while it rolls.

jumper.cartwheel_ten's task (reference 5, straight) plus `legs_straight` in the roll:
each limb fully extended hip to tip (full_mdp.leg_shortening; a 0.3 rad knee bend
costs most of it). The hips stay free: swinging a straight leg forward to plant
ahead, and tilting it out of the wheel's plane (shortening its reach in the plane)
are the movements left to keep the wheel going. Run 3's policy rolled 10+ turns with
every joint swinging 20-40 deg (up to ~100 deg), knees included.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_ten_straight",
    assets=JUMPER_ASSETS,
    description="jumper hexapod 10-turn straight cartwheel with the legs kept straight while rolling (cartwheel_ten.npz + legs_straight)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
