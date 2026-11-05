"""jumper.cartwheel_stunt_fast -- cartwheel down a hill, up a kicker, over a 0.45 m gap
and onto a landing ramp, legs free to move all the way to roll faster
(tasks/jumper/cartwheel_stunt)."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_stunt_fast",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel ramp jump over a gap, legs free to roll faster",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
