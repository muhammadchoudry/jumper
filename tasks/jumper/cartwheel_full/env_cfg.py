"""jumper.cartwheel_full -- `jumper.cartwheel_jump`'s scene, reference and spawn
scheme, with each reward and termination switched on only in its phase (see
`full_mdp.py`):

A (stand, run-up, jump, landing): the jump task's imitation terms, the full
  rotation tracking, and its divergence terminations.
B (roll): `jumper.cartwheel_roll`'s terms -- roll about the world axle, forward
  travel, on edge, straight, light pull toward the star; ends if it topples off
  its edge (after the landing has had 0.4 s to stand up) or the hub drops below
  0.15 m.
C (tip down, stand): joint pose, height and gravity direction tracked toward the
  reference's stand, plus upright and settle; nothing ends it but the clock.

Residual scale 0.7 rad as in jumper.cartwheel_roll, so that policy's checkpoint is
a working start for phase B (same observation and action spaces).
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel_jump.env_cfg import JUMP_REF_NPZ
from ..cartwheel_jump.env_cfg import env_cfg as _cartwheel_jump_env_cfg
from ..cartwheel_roll import roll_mdp
from . import full_mdp as F

ACTION_SCALE = 0.7


def _gate(term: RewardTermCfg, phases: str, after_start: float = 0.0, weight: float | None = None,
          **params) -> RewardTermCfg:
    return RewardTermCfg(
        func=F.gated, weight=term.weight if weight is None else weight,
        params={"fn": term.func, "phases": phases, "after_start": after_start, **term.params, **params},
    )


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_jump_env_cfg(asset=asset, play=play)
    act = cfg.actions["joint_pos"]
    act.scale = {j: ACTION_SCALE for j in act.scale}

    r = cfg.rewards
    rewards = {
        # A: track the run-up and jump; C keeps pose/height/gravity-direction tracking for the stand
        "ref_joint_pos": _gate(r["ref_joint_pos"], "AC"),
        "ref_base_height": _gate(r["ref_base_height"], "AC"),
        "ref_base_ori": _gate(r["ref_base_ori"], "AC"),
        "ref_base_lin_vel": _gate(r["ref_base_lin_vel"], "A"),
        "ref_foot_pos": _gate(r["ref_foot_pos"], "A"),
        "ref_joint_vel": _gate(r["ref_joint_vel"], "A"),
        "ref_base_rotation": _gate(r["ref_base_rotation"], "A"),
        # B: roll freely
        "roll_forward": RewardTermCfg(func=F.gated, weight=3.0,
                                      params={"fn": roll_mdp.roll_forward, "phases": "B"}),
        "travel_forward": RewardTermCfg(func=F.gated, weight=3.0,
                                        params={"fn": roll_mdp.travel_forward, "phases": "B"}),
        "on_edge": RewardTermCfg(func=F.gated, weight=1.0, params={"fn": roll_mdp.on_edge, "phases": "B"}),
        "straight": RewardTermCfg(func=F.gated, weight=0.5, params={"fn": roll_mdp.straight, "phases": "B"}),
        "star_pose": _gate(r["ref_joint_pos"], "B", weight=0.3, sigma_sq=1.0),
        # C: back on its feet and still
        "upright": RewardTermCfg(func=F.gated, weight=2.0, params={"fn": F.upright, "phases": "C"}),
        "settle": RewardTermCfg(func=F.gated, weight=1.0, params={"fn": F.settle, "phases": "C"}),
        # always
        "alive": r["alive"],
        "dof_pos_limits": r["dof_pos_limits"],
        "undesired_contacts": r["undesired_contacts"],
        "action_rate_l2": r["action_rate_l2"],
    }
    rewards["undesired_contacts"].weight = -2.0
    rewards["action_rate_l2"].weight = -0.01 * (0.30 / ACTION_SCALE) ** 2
    cfg.rewards = rewards

    t = cfg.terminations
    cfg.terminations = {
        "time_out": t["time_out"],
        "rotation_diverged": TerminationTermCfg(
            func=F.gated_done, params={"fn": t["rotation_diverged"].func, "phases": "A",
                                       **t["rotation_diverged"].params}),
        "orientation_diverged": TerminationTermCfg(
            func=F.gated_done, params={"fn": t["orientation_diverged"].func, "phases": "A",
                                       **t["orientation_diverged"].params}),
        "fell_off_edge": TerminationTermCfg(
            func=F.gated_done, params={"fn": roll_mdp.fell_off_edge, "phases": "B", "after_start": 0.4,
                                       "max_tilt_cos": 0.7}),
        "hub_too_low": TerminationTermCfg(
            func=F.gated_done, params={"fn": roll_mdp.hub_too_low, "phases": "B", "min_height": 0.15}),
    }
    if "reference_diverged" in t:
        cfg.terminations["reference_diverged"] = TerminationTermCfg(
            func=F.gated_done, params={"fn": t["reference_diverged"].func, "phases": "A",
                                       **t["reference_diverged"].params})

    # reference spans 5.4 s after go (+0.3 s stand before it), plus time to settle
    cfg.episode_length_s = 6.4
    cfg.viewer.distance = 2.4
    return cfg


__all__ = ["env_cfg", "JUMP_REF_NPZ"]
