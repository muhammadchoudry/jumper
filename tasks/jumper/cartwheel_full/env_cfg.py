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

import math

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
STRAIGHT_WEIGHT = 3.0
EVEN_WEIGHT = 3.0


def _gate(term: RewardTermCfg, phases: str, after_start: float = 0.0, weight: float | None = None,
          **params) -> RewardTermCfg:
    return RewardTermCfg(
        func=F.gated, weight=term.weight if weight is None else weight,
        params={"fn": term.func, "phases": phases, "after_start": after_start, **term.params, **params},
    )


#: roll-phase rewards (roll, travel, lean, heading) per steering mode:
#:   False       straight along world +x (cartwheel_roll's terms)
#:   True/"ref"  follow the reference's heading at each moment (figure 8 tries 1-2)
#:   "distance"  draw a figure 8 by distance rolled, from wherever the roll starts,
#:               mirrored in half the envs (figure 8 try 3)
B_FN = {
    False: (roll_mdp.roll_forward, roll_mdp.travel_forward, roll_mdp.on_edge, roll_mdp.straight),
    True: (F.steer_roll, F.steer_travel, F.steer_lean, F.steer_heading),
    "ref": (F.steer_roll, F.steer_travel, F.steer_lean, F.steer_heading),
    "distance": (F.eight_roll, F.eight_travel, F.eight_lean, F.eight_heading),
}


def env_cfg(asset: Path | None = None, play: bool = False):
    return build(asset=asset, play=play)


def build(asset: Path | None = None, play: bool = False, motion: Path | None = None,
          steer: bool | str = False, episode_s: float = 6.4, straight_legs: bool | str = False,
          slope_deg: float = 0.0, even_weight: float | None = None, even_sigma: float = 0.35):
    """The task, optionally on another reference-5-style motion (`motion`), with the
    roll scored against the reference's heading and turn (`steer`; for a path that
    curves, e.g. a figure 8), and an episode of `episode_s` seconds."""
    cfg = _cartwheel_jump_env_cfg(asset=asset, play=play)
    if motion is not None:
        cfg.commands["jump"].motion_file = str(motion)

        def _reference_contract() -> dict:
            from ..cartwheel import mdp as cw_mdp
            from ..common.constants import HOME
            from ..jump.env_cfg import _DECIMATION, _FUTURE_OFFSETS, _SIM_TIMESTEP
            ref = cw_mdp.reference.JumpReference(motion, "cpu", list(HOME))
            return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION, starts_on_entry=True)

        cfg.reference_contract = _reference_contract
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
                                      params={"fn": B_FN[steer][0], "phases": "B"}),
        "travel_forward": RewardTermCfg(func=F.gated, weight=3.0,
                                        params={"fn": B_FN[steer][1], "phases": "B"}),
        "on_edge": RewardTermCfg(func=F.gated, weight=1.0, params={"fn": B_FN[steer][2], "phases": "B"}),
        "straight": RewardTermCfg(func=F.gated, weight=0.5 if not steer else 1.5,
                                  params={"fn": B_FN[steer][3], "phases": "B"}),
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
    if straight_legs:
        # legs fully extended hip to tip while rolling; the hips stay free to swing a
        # straight leg forward and tilt it out of the wheel's plane (its reach in the
        # plane shortens -- the pumping a bent knee did), so the old pull toward the
        # star's exact joint angles is relaxed
        rewards["legs_straight"] = RewardTermCfg(func=F.gated, weight=STRAIGHT_WEIGHT,
                                                 params={"fn": F.legs_straight, "phases": "B", "after_start": 0.3})
        rewards["star_pose"] = _gate(r["ref_joint_pos"], "B", weight=0.3, sigma_sq=1.0)
    if straight_legs == "even":
        # straight AND evenly spread: each tip held on its star spoke, 60 deg apart in
        # the wheel's plane (spokes_even); only tilting a straight leg out of the plane
        # is left free. Kernels wide enough to give credit from where the policy starts
        # (legs ~3.5% short, spokes ~35 deg rms off even)
        rewards["legs_straight"].params["sigma"] = 0.05
        rewards["spokes_even"] = RewardTermCfg(func=F.gated, weight=even_weight or EVEN_WEIGHT,
                                               params={"fn": F.spokes_even, "phases": "B", "after_start": 0.3,
                                                       "sigma": even_sigma})
        rewards.pop("star_pose")
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
        if straight_legs == "even":     # the straight-legs run forgot the trot mid-run
            ev.params.update(focus_fraction=0.25, start_fraction=0.25)

    # reference spans 5.4 s after go (+0.3 s stand before it), plus time to settle
    cfg.episode_length_s = episode_s
    if slope_deg:
        # downhill toward +x: gravity tilted instead of the floor (the same physics; the
        # floor stays z = 0, so the reference's positions and heights still hold)
        a = math.radians(slope_deg)
        cfg.sim.mujoco.gravity = (9.81 * math.sin(a), 0.0, -9.81 * math.cos(a))
    cfg.viewer.distance = 2.4
    return cfg


__all__ = ["build", "env_cfg", "JUMP_REF_NPZ"]
