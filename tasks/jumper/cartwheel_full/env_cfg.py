"""jumper.cartwheel_full -- `jumper.cartwheel_jump`'s scene, reference and spawn
scheme, with each reward and termination switched on only in its phase (see
`full_mdp.py`):

A (stand, run-up, jump, landing): the jump task's imitation terms and full
  rotation tracking. Terminations: the full-rotation and gravity checks (1 rad)
  during the run-up, a looser full-rotation check (1.6 rad) during the jump, joint
  divergence to touch-down, and none after it.
B (roll): `jumper.cartwheel_roll`'s terms -- roll about the world axle, forward
  travel, on edge, straight -- and a firm pull toward the spread-out star
  (`star_pose`, weight 1.5, kernel 0.15 rad^2 mean-square: ~0.15 rad of pumping
  costs little, folded legs cost most of it); ends if it topples off
  its edge (after the landing has had 0.4 s to stand up) or the hub drops below
  0.15 m (both after 0.4 s to recover from the landing).
C (tip down, stand): joint pose, height and gravity direction tracked toward the
  reference's stand, plus upright and settle; nothing ends it but the clock.

Residual reach 0.7 rad in the roll (as jumper.cartwheel_roll, whose checkpoint is the
starting point: same observation and action spaces) and 0.4 rad elsewhere
(`full_actions.py`): one 0.7 everywhere let the rolling policy's large outputs throw
the trot off the reference in its first second.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel_jump.env_cfg import JUMP_REF_NPZ
from ..cartwheel_jump.env_cfg import env_cfg as _cartwheel_jump_env_cfg
from ..cartwheel_roll import roll_mdp
from . import full_mdp as F
from .full_actions import PhaseScaledResidualActionCfg

ACTION_SCALE = 0.7
JUMP_MAX_ROTATION_ERROR = 1.6   # rad, during the jump only
STAR_WEIGHT, STAR_SIGMA_SQ = 1.5, 0.15


def _gate(term: RewardTermCfg, phases: str, after_start: float = 0.0, weight: float | None = None,
          **params) -> RewardTermCfg:
    return RewardTermCfg(
        func=F.gated, weight=term.weight if weight is None else weight,
        params={"fn": term.func, "phases": phases, "after_start": after_start, **term.params, **params},
    )


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _cartwheel_jump_env_cfg(asset=asset, play=play)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = PhaseScaledResidualActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names,
        scale={j: ACTION_SCALE for j in old.scale}, command_name=old.command_name,
    )

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
        "star_pose": _gate(r["ref_joint_pos"], "B", weight=STAR_WEIGHT, sigma_sq=STAR_SIGMA_SQ),
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
    rot, ori = t["rotation_diverged"], t["orientation_diverged"]
    cfg.terminations = {
        "time_out": t["time_out"],
        # the run-up: the full-rotation and gravity checks as in cartwheel_jump
        "rotation_diverged": TerminationTermCfg(
            func=F.gated_window, params={"fn": rot.func, "end": "jump", **rot.params}),
        "orientation_diverged": TerminationTermCfg(
            func=F.gated_window, params={"fn": ori.func, "end": "jump", **ori.params}),
        # the jump: looser (the robot tips onto its edge its own way), and nothing after
        # touch-down -- round 8899 lost 13/16 attempts to the 1 rad check in the 0.15 s
        # after landing, leaning more than the reference but able to roll on
        "rotation_diverged_jump": TerminationTermCfg(
            func=F.gated_window, params={"fn": rot.func, "start": "jump", "end": "roll",
                                         **{**rot.params, "max_angle": JUMP_MAX_ROTATION_ERROR}}),
        "fell_off_edge": TerminationTermCfg(
            func=F.gated_done, params={"fn": roll_mdp.fell_off_edge, "phases": "B", "after_start": 0.4,
                                       "max_tilt_cos": 0.7}),
        "hub_too_low": TerminationTermCfg(
            func=F.gated_done, params={"fn": roll_mdp.hub_too_low, "phases": "B", "after_start": 0.4,
                                       "min_height": 0.15}),
    }
    if "reference_diverged" in t:
        cfg.terminations["reference_diverged"] = TerminationTermCfg(
            func=F.gated_window, params={"fn": t["reference_diverged"].func, "end": "roll",
                                         **t["reference_diverged"].params})

    # a third of the practice starts around the jump and landing
    ev = cfg.events["reset_from_reference"]
    if not play:
        ev.func = F.reset_focus
        ev.params = {**ev.params, "focus_s": (-0.3, 0.4), "focus_fraction": 0.35}

    # reference spans 5.4 s after go (+0.3 s stand before it), plus time to settle
    cfg.episode_length_s = 6.4
    cfg.viewer.distance = 2.4
    return cfg


__all__ = ["env_cfg", "JUMP_REF_NPZ"]
