"""Rewards, terminations and spawns for the stunt course (course.py)."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..cartwheel_full import full_mdp as F
from . import course

_ROBOT = SceneEntityCfg("robot")


def _c(env) -> course.Course:
    return getattr(env, "_course", course.DEFAULT)


def _hub(env):
    return env.scene["robot"].data.root_link_pos_w


def reset_course(env, env_ids, course_name: str = "default", **kw) -> None:
    """reset_focus, plus which course and where its roll ends (phase C by place,
    full_mdp.phase_mask)."""
    env._course = course.COURSES[course_name]
    env._stand_x = env._course.STAND_X
    F.reset_focus(env, env_ids, **kw)


def hub_low_course(env, min_height: float = 0.15) -> torch.Tensor:
    """Hub less than min_height above the course under it (legs folded / tumbling)."""
    p = _hub(env)
    return p[:, 2] - _c(env).surface_z(p[:, 0]) < min_height


def fell_in_gap(env) -> torch.Tensor:
    """Hub below the landing lip while over the gap or short of the landing ramp."""
    p = _hub(env)
    over = (p[:, 0] > _c(env).LIP_X - 0.15) & (p[:, 0] < _c(env).LAND_X + 0.25)
    return over & (p[:, 2] < _c(env).LAND_Z - 0.15)


def off_course(env) -> torch.Tensor:
    return _hub(env)[:, 1].abs() > _c(env).half_width - 0.1


def cleared_gap(env) -> torch.Tensor:
    """1 on the step the hub passes 0.2 m beyond the landing lip, above the ramp."""
    p = _hub(env)
    x = p[:, 0]
    prev = getattr(env, "_course_prev_x", None)
    if prev is None or prev.shape != x.shape:
        prev = x.clone()
    line = _c(env).LAND_X + 0.2
    hit = (prev < line) & (x >= line) & (p[:, 2] - _c(env).surface_z(x) > 0.1)
    # a reset puts x back near the start: never counts as a crossing
    env._course_prev_x = x.clone()
    return hit.float()


def centre_line(env, sigma: float = 0.5) -> torch.Tensor:
    """1 on the course's centre line (y = 0), falling off sideways against sigma (m)."""
    y = _hub(env)[:, 1]
    return torch.exp(-(y * y) / (sigma * sigma))


def outside_x(env, term, x0: float, x1: float, **params) -> torch.Tensor:
    """A reward `fn` paid only while the hub is outside [x0, x1] along the course
    (inside: near the lip, where the legs are free to pop)."""
    x = _hub(env)[:, 0]
    return term(env, **params) * ((x < x0) | (x > x1)).float()


__all__ = ["centre_line", "cleared_gap", "fell_in_gap", "hub_low_course", "off_course", "outside_x", "reset_course"]
