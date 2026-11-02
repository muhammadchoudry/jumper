"""jumper.cartwheel_runup -- `jumper.cartwheel`'s training scheme on reference 3, with
more room for the policy.

Changes from `jumper.cartwheel_flat`, from what its plateau showed (it tipped onto
its edge but could not start or keep the wheel rolling, and toppled sideways):

- **Reference 3**: a running start, so the wheel arrives already turning.
- **Residual scale 0.25 -> 0.4 rad**: a balancing wheel steers with out-of-plane
  limb movements the reference does not contain.
- **Orientation cutoff 1.0 -> 1.4 rad**: let wobbling attempts continue long
  enough to learn the recovery.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel import mdp as cw_mdp
from ..cartwheel.env_cfg import env_cfg as _cartwheel_env_cfg
from ..jump.env_cfg import _DECIMATION, _FUTURE_OFFSETS, _SIM_TIMESTEP

RUNUP_REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_runup.npz"
ACTION_SCALE = 0.4
MAX_ORIENTATION_ERROR = 1.4
MAX_ROTATION_ERROR = 1.0


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_env_cfg(asset=asset, play=play)
    cfg.commands["jump"].motion_file = str(RUNUP_REF_NPZ)
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
    # span 5.0 s after go plus a settling margin
    cfg.episode_length_s = 5.6
    cfg.viewer.distance = 2.2

    def _reference_contract() -> dict:
        from ..common.constants import HOME
        ref = cw_mdp.reference.JumpReference(RUNUP_REF_NPZ, "cpu", list(HOME))
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=True)

    cfg.reference_contract = _reference_contract
    return cfg


__all__ = ["env_cfg", "RUNUP_REF_NPZ"]
