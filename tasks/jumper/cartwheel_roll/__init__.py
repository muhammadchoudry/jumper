"""jumper.cartwheel_roll -- the flat-crab wheel, legs free: learn to keep rolling.

The rigid six-spoke star cannot coast (it loses ~2/3 of its spin energy at every
spoke change and stops ~25 deg after a 2.4 rad/s start, measured in
tools/crab_trio/limb_wheel.py), and a hand-scripted stepping gait got only ~1/3
turn. Here nothing prescribes the legs: the policy starts on edge already rolling
(spawn states `ref/roll_loop.npz`, `tools/crab_trio/roll_loop_ref.py`) and is
paid for turning the wheel forward while staying up on its edge, shell off the
floor, rolling straight. What its legs learn to do becomes the basis of the
final cartwheel reference.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_roll",
    assets=JUMPER_ASSETS,
    description="jumper hexapod rolling on edge as a six-spoke wheel, legs free: rewarded for wheel turning, staying on edge, shell off the floor (roll_loop.npz spawn states)",
    tags=("locomotion", "jumper", "cartwheel"),
)
