"""jumper.cartwheel_stunt_pop / _fast: the full cartwheel -- trot, jump, land in the
star -- then over the plateau's edge, down a 15 deg hill, up a kicker, over a 0.45 m
gap, onto a landing ramp, and up on its feet on the run-out (course.py;
docs/jumper_stunt_plan.md).

Both on `cartwheel_full.env_cfg.build` with cartwheel_ten's reference (the run-up
and jump happen on the plateau). Changes for the course:
- the scene: course boxes on the terrain body, the ground plane dropped to the bottom
  of the pit; env_spacing 0 (every env is its own MjData, on its own course);
- the roll ends by place too: past STAND_X (2 m onto the run-out) it is phase C
  (stand up), whatever the clock says; the reference's pose and height are only
  tracked in A (its stand happens at its own time and height);
- rolling and travel rewards uncapped up to 10 rad/s and 3 m/s (the gap needs ~2);
- terminations: hub less than 0.15 m above the course (not the world), fell in the
  gap, off the side; a bonus for clearing the gap;
- spawns only before the plateau's edge (standing, the jump, the landing).

Variants:
- **pop**: straight, evenly spread legs (cartwheel_hill's terms) everywhere except
  from 0.45 m before the kicker's lip to the landing lip, where the legs are free --
  to push off the lip (a radial leg push as the last spoke leaves it).
- **fast**: legs free to move all the way down (only a light pull to straight legs),
  to roll faster than a rigid star can.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..cartwheel_full import full_mdp as F
from ..cartwheel_full.env_cfg import build
from ..cartwheel_ten.env_cfg import REF_NPZ
from . import course, stunt_mdp as S

EPISODE_S = 9.8


def _spawn_end_fraction() -> float:
    """Reference phase at 0.4 s into the roll (still on the plateau)."""
    D = np.load(REF_NPZ)
    meta = json.loads(str(D["meta"]))
    t = float(meta["t_go"]) + float(meta["segments_s_after_go"]["roll"][0]) + 0.4
    return t / float(D["t"][-1])


def build_stunt(asset: Path | None, play: bool, variant: str):
    cfg = build(asset=asset, play=play, motion=REF_NPZ, steer=False, episode_s=EPISODE_S, straight_legs="even",
                even_weight=5.0, even_sigma=0.25)
    cfg.scene.env_spacing = 0.0
    cfg.scene.spec_fn = course.spec_fn

    R = cfg.rewards
    for k in ("ref_joint_pos", "ref_base_height", "ref_base_ori"):
        R[k].params["phases"] = "A"
    R["roll_forward"].params["max_rate"] = 10.0
    R["travel_forward"].params["max_speed"] = 3.0
    if variant == "pop":
        x0, x1 = course.LIP_X - 0.45, course.LAND_X + 0.1
        for k in ("legs_straight", "spokes_even"):
            p = R[k].params
            inner = {kk: v for kk, v in p.items() if kk not in ("fn", "phases", "after_start")}
            R[k] = RewardTermCfg(func=F.gated, weight=R[k].weight,
                                 params={"fn": S.outside_x, "phases": p["phases"], "after_start": p["after_start"],
                                         "term": p["fn"], "x0": x0, "x1": x1, **inner})
    elif variant == "fast":
        R.pop("spokes_even")
        R["legs_straight"].weight = 0.5
    else:
        raise ValueError(variant)
    R["cleared_gap"] = RewardTermCfg(func=S.cleared_gap, weight=1000.0)

    T = cfg.terminations
    T["hub_too_low"] = TerminationTermCfg(func=F.gated_done, params={"fn": S.hub_low_course, "phases": "B",
                                                                     "after_start": 0.4, "min_height": 0.15})
    T["fell_in_gap"] = TerminationTermCfg(func=S.fell_in_gap)
    T["off_course"] = TerminationTermCfg(func=S.off_course)

    ev = cfg.events["reset_from_reference"]
    ev.func = S.reset_course
    if not play:
        ev.params["phase_range"] = (0.0, _spawn_end_fraction())
    cfg.viewer.distance = 3.0
    return cfg
