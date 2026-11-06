"""jumper.cartwheel_stunt_long -- the roll-faster ramp jump over a 0.75 m gap,
wider than the crab is across (course.LONG: an 18 deg hill, the landing lip 0.4 m
below the kicker's, a 45 deg landing ramp easing to 20 deg). Legs free to roll
fast; a pull back to the course's centre line on the way down. Warm-started from
jumper.cartwheel_stunt_fast (fast_v1_model_22300).

Sizing: run 1 left the kicker lip at ~2.2 m/s with the hub ~0.21 m above it and
almost no upward speed (the 14 cm kicker is shorter than one spoke step, so it
barely turns the wheel upward). From there the hub clears a gap G when
0.21 + drop - g/2 (G/v)^2 >= ~0.1: G <= 0.46 m at drop 0.1 (the 0.45 m course),
0.71 m at drop 0.4 -- so the extra height comes from the drop, and the steeper
hill buys speed margin.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_stunt_long",
    assets=JUMPER_ASSETS,
    description="jumper hexapod cartwheel ramp jump over a 0.75 m gap, legs free to roll faster",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
