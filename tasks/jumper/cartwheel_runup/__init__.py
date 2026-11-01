"""jumper.cartwheel_runup -- reference 3: flat-crab cartwheel with a running start.

The crab trots forward (a physics recording of the trained `jumper.posture`
walking policy), tips onto its edge while still moving, spreads into the flat
six-spoke star and rolls one full turn, then stands. The run-up gives the wheel
the momentum reference 2 lacked when it started from a dead stop on its edge.
Reference `ref/cartwheel_runup.npz` from `tools/crab_trio/cartwheel_ref3.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_runup",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-crab cartwheel with a running start: trot, tip onto edge, roll 360 on the six-spoke star (cartwheel_runup.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
