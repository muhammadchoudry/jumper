"""jumper.cartwheel_flat -- reference 2: the flat-crab cartwheel.

All six limbs straight and spread in the plane of the body (a six-spoke star,
60 degrees apart); the crab tips onto its edge and rolls 360 degrees
head-over-tail like a wheel, then tips back onto its feet. Reference
`ref/cartwheel_flat_crab.npz` from `tools/crab_trio/cartwheel_ref2.py`; the
training scheme is `jumper.cartwheel`'s.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel_flat",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-crab cartwheel: six straight limbs spread in the body plane, rolling on edge (cartwheel_flat_crab.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
