"""jumper.cartwheel_stunt_pop -- cartwheel down a hill, up a kicker, over a 0.45 m gap
and onto a landing ramp, legs held in the star except near the lip, where they are
free to push off (tasks/jumper/cartwheel_stunt)."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_stunt_pop",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel ramp jump over a gap, star legs, pushing off the lip",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
