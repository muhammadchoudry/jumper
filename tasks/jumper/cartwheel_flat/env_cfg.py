"""jumper.cartwheel_flat -- `jumper.cartwheel`'s training scheme on reference 2.

Everything (spin-keeping spawn, orientation-divergence termination, claws as
feet, no symmetry) is the cartwheel task's; only the reference changes. See
`tasks/jumper/cartwheel/env_cfg.py`.
"""

from __future__ import annotations

from pathlib import Path

from ..cartwheel import mdp as cw_mdp
from ..cartwheel.env_cfg import env_cfg as _cartwheel_env_cfg
from ..jump.env_cfg import _DECIMATION, _FUTURE_OFFSETS, _SIM_TIMESTEP

FLAT_CRAB_REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_flat_crab.npz"


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_env_cfg(asset=asset, play=play)
    cfg.commands["jump"].motion_file = str(FLAT_CRAB_REF_NPZ)
    # span 3.0 s after go plus a settling margin
    cfg.episode_length_s = 3.6
    cfg.viewer.distance = 2.0

    def _reference_contract() -> dict:
        from ..common.constants import HOME
        ref = cw_mdp.reference.JumpReference(FLAT_CRAB_REF_NPZ, "cpu", list(HOME))
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=True)

    cfg.reference_contract = _reference_contract
    return cfg


__all__ = ["env_cfg", "FLAT_CRAB_REF_NPZ"]
