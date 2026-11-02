"""jumper.cartwheel_jump -- `jumper.cartwheel_runup`'s training scheme (rotation
termination + full-rotation reward, residual scale 0.4) on reference 4, where a
sideways jump after the trot lifts the hub onto the first spoke already rolling.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel import mdp as cw_mdp
from ..cartwheel.env_cfg import env_cfg as _cartwheel_env_cfg
from ..jump.env_cfg import _DECIMATION, _FUTURE_OFFSETS, _SIM_TIMESTEP

JUMP_REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_jump.npz"
ACTION_SCALE = 0.4
MAX_ORIENTATION_ERROR = 1.4
MAX_ROTATION_ERROR = 1.0


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_env_cfg(asset=asset, play=play)
    cfg.commands["jump"].motion_file = str(JUMP_REF_NPZ)
    act = cfg.actions["joint_pos"]
    act.scale = {j: ACTION_SCALE for j in act.scale}
    cfg.terminations["orientation_diverged"].params["max_angle"] = MAX_ORIENTATION_ERROR
    # ...but the gravity-only check cannot see the wheel turning: round 1700 learned to
    # lie ~70 deg off its edge and wait out the roll. End attempts whose full rotation
    # strays from the reference's, and pay for following the full rotation.
    cfg.terminations["rotation_diverged"] = TerminationTermCfg(
        func=cw_mdp.terminations.rotation_diverged,
        params={"command_name": "jump", "max_angle": MAX_ROTATION_ERROR},
    )
    cfg.rewards["ref_base_rotation"] = RewardTermCfg(
        func=cw_mdp.cw_rewards.ref_base_rotation, weight=2.0,
        params={"command_name": "jump", "sigma_sq": 0.15},
    )
    # span 5.2 s after go plus a settling margin
    cfg.episode_length_s = 6.0
    cfg.viewer.distance = 2.2

    def _reference_contract() -> dict:
        from ..common.constants import HOME
        ref = cw_mdp.reference.JumpReference(JUMP_REF_NPZ, "cpu", list(HOME))
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=True)

    cfg.reference_contract = _reference_contract
    return cfg


__all__ = ["env_cfg", "JUMP_REF_NPZ"]
