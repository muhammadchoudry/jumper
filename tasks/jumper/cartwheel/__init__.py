"""jumper.cartwheel -- reference-guided sideways cartwheel on straight-limb spokes.

The robot crouches with its legs fanned, pushes off sideways, snaps its limbs out
straight into a six-spoke "star" and rolls 360 degrees over them -- legs, claws
over the top, legs -- back onto its feet. The reference
(`ref/cartwheel_flat.npz`, authored by `tools/cartwheel/refs/cartwheel_ref.py`) is
kinematic; the policy learns residual corrections that make it work under
physics. Built on `jumper.jump`'s reference-tracking scheme; see `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.cartwheel",
    assets=JUMPER_ASSETS,
    description="jumper hexapod reference-guided sideways cartwheel on straight-limb spokes (cartwheel_flat.npz)",
    tags=("locomotion", "jumper", "cartwheel", "reference-tracking"),
)
