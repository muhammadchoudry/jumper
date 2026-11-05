"""jumper.cartwheel_hill -- the cartwheel down an 8 deg hill, holding the star.

On flat ground a straight, evenly spread star cannot keep rolling (~2/3 of its
spin energy goes at each spoke change), so cartwheel_ten_star's policy bunched its
legs to pivot foot to foot. Downhill, each spoke change also drops the hub by
~0.31 m * sin(slope), which pays for that loss, so the legs can stay a star; the
policy's job becomes balance (a thin wheel on single tips falls sideways in under a
second, scripted: tools/crab_trio/hill_roll.py) and not buckling on landing.

Same rewards as cartwheel_ten_star with spokes_even stronger (weight 5, kernel
0.25 rad). The slope is gravity tilted toward +x; traj_video.py renders a real hill
(SLOPE_DEG=8). Warm-started from cartwheel_ten_star (star_v1_model_18896).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_hill",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel down an 8 deg hill, legs held in the star (straight, evenly spread)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
