"""jumper.cartwheel_roll -- `jumper.cartwheel`'s scene and spawn scheme with the legs
set free and the reward on rolling, not on copying a motion.

From `jumper.cartwheel`:
- spawn states from the reference with their spin (`ref/roll_loop.npz`: the star on
  edge rolling at 2.7 rad/s), the phase observation, residual actions around the
  star pose, undesired-contact sensor on chassis + upper limbs.

Changed:
- **Residual scale 0.7 rad** (was 0.25-0.4): room for stepping, folding, pumping.
- **Rewards**: wheel turning forward (about the world axle, see `roll_mdp.roll_forward`)
  and hub travelling forward (weight 3; both only while on edge), staying on edge, rolling straight; a light pull toward the star pose
  (`ref_joint_pos`, weight 0.3, wide kernel) so the legs stay spokes, not a crawl;
  shell/upper-limb ground contact -2. All other imitation terms dropped.
- **Terminations**: toppled more than ~44 deg off the edge, hub below 0.15 m (legs
  folded, tumbling on its knees), or time out. No
  reference-divergence terminations -- diverging is the point.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel import mdp as cw_mdp
from ..cartwheel.env_cfg import env_cfg as _cartwheel_env_cfg
from ..jump.env_cfg import _DECIMATION, _FUTURE_OFFSETS, _SIM_TIMESTEP
from . import roll_mdp

ROLL_REF_NPZ = Path(__file__).resolve().parent / "ref" / "roll_loop.npz"
ACTION_SCALE = 0.7


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_env_cfg(asset=asset, play=play)
    cfg.commands["jump"].motion_file = str(ROLL_REF_NPZ)
    act = cfg.actions["joint_pos"]
    act.scale = {j: ACTION_SCALE for j in act.scale}

    keep = {k: cfg.rewards[k] for k in ("alive", "dof_pos_limits", "undesired_contacts", "action_rate_l2")
            if k in cfg.rewards}
    keep["undesired_contacts"].weight = -2.0
    keep["action_rate_l2"].weight = -0.01 * (0.30 / ACTION_SCALE) ** 2
    ref_pos = cfg.rewards["ref_joint_pos"]
    ref_pos.weight = 0.3
    ref_pos.params["sigma_sq"] = 1.0
    cfg.rewards = {
        **keep,
        "ref_joint_pos": ref_pos,
        "roll_forward": RewardTermCfg(func=roll_mdp.roll_forward, weight=3.0),
        "travel_forward": RewardTermCfg(func=roll_mdp.travel_forward, weight=3.0),
        "on_edge": RewardTermCfg(func=roll_mdp.on_edge, weight=1.0),
        "straight": RewardTermCfg(func=roll_mdp.straight, weight=0.5),
    }

    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "fell_off_edge": TerminationTermCfg(func=roll_mdp.fell_off_edge, params={"max_tilt_cos": 0.7}),
        "hub_too_low": TerminationTermCfg(func=roll_mdp.hub_too_low, params={"min_height": 0.15}),
    }

    # spawn anywhere in the first 2.5 turns of the roll
    cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.8) if not play else (0.0, 0.01)
    cfg.episode_length_s = 5.0
    cfg.viewer.distance = 2.2

    def _reference_contract() -> dict:
        from ..common.constants import HOME
        ref = cw_mdp.reference.JumpReference(ROLL_REF_NPZ, "cpu", list(HOME))
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=True)

    cfg.reference_contract = _reference_contract
    return cfg


__all__ = ["env_cfg", "ROLL_REF_NPZ"]
