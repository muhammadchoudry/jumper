"""jumper.cartwheel_jump -- reference 4: flat-crab cartwheel launched by a sideways jump.

The crab trots forward (recorded `jumper.posture` walk), crouches and jumps
sideways -- left legs drive harder -- snapping into the flat six-spoke star in
the air and landing on its right edge already rolling forward. One full turn,
then it stands. The jump supplies the ~4 J of lift that tipping up from the
trot alone (references 2 and 3) could not.
Reference `ref/cartwheel_jump.npz` from `tools/crab_trio/cartwheel_ref4.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_jump",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-crab cartwheel: trot, sideways jump into the six-spoke star, land rolling, roll 360, stand (cartwheel_jump.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
