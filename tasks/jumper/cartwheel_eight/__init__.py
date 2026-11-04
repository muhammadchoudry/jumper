"""jumper.cartwheel_eight -- the full cartwheel, rolling a figure 8.

`jumper.cartwheel_full`'s phase-gated task (cartwheel_full/env_cfg.py `build`) on
reference 5 (`ref/cartwheel_eight.npz`, tools/crab_trio/cartwheel_ref5.py): trot, sideways
jump, a roll segment ~5.5 s long (about 10 turns at the policy's pace), tip down,
stand. Try 3: each env draws its own figure 8 by distance rolled from where its roll begins (full_mdp.eight_state), left loop first or right loop first at random; heading and lean (tan(lean) = v^2 * curvature / g, as a rolling coin leans) are scored against that.
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
