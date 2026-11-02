"""Phase gating for jumper.cartwheel_full.

Phases come from the reference's own segment table (meta["segments_s_after_go"]),
read once, and the time since go each env is at:

  A = before the roll starts + LANDING (waiting stand, run-up, jump, touch-down)
  B = the rest of the roll
  C = tip-down and stand-end, to the end of the episode
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..jump.mdp.reference import get_reference, ref_state

LANDING = 0.15          # s into the roll segment still scored as the jump's landing
_ROBOT = SceneEntityCfg("robot")


def _bounds(env) -> tuple[float, float]:
    if not hasattr(env, "_cw_full_bounds"):
        seg = get_reference(env).meta["segments_s_after_go"]
        env._cw_full_bounds = (float(seg["roll"][0]) + LANDING, float(seg["roll"][1]))
    return env._cw_full_bounds


def phase_mask(env, phases: str, command_name: str = "jump", after_start: float = 0.0) -> torch.Tensor:
    """1.0 where the env is in one of `phases` (any of "ABC"); `after_start` s
    trims the start of phase B (for terms that should wait for the landing to settle)."""
    _, _, _, tsg = ref_state(env, command_name)
    b0, b1 = _bounds(env)
    m = torch.zeros_like(tsg, dtype=torch.bool)
    if "A" in phases:
        m |= tsg < b0
    if "B" in phases:
        m |= (tsg >= b0 + after_start) & (tsg < b1)
    if "C" in phases:
        m |= tsg >= b1
    return m


def gated(env, fn, phases: str, after_start: float = 0.0, **params) -> torch.Tensor:
    """A reward term `fn` counted only in `phases`."""
    return fn(env, **params) * phase_mask(env, phases, after_start=after_start).float()


def gated_done(env, fn, phases: str, after_start: float = 0.0, **params) -> torch.Tensor:
    """A termination `fn` that only ends episodes in `phases`."""
    return fn(env, **params) & phase_mask(env, phases, after_start=after_start)


def settle(env, sigma_sq: float = 0.1, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 when the body is still (linear and angular speed small)."""
    robot: Entity = env.scene[asset_cfg.name]
    v = (robot.data.root_link_lin_vel_w ** 2).sum(dim=1) + 0.05 * (robot.data.root_link_ang_vel_w ** 2).sum(dim=1)
    return torch.exp(-v / sigma_sq)


def upright(env, sigma_sq: float = 0.1, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 when standing on its feet (gravity straight down the body's -z)."""
    robot: Entity = env.scene[asset_cfg.name]
    e = (robot.data.projected_gravity_b[:, 2] + 1.0) ** 2
    return torch.exp(-e / sigma_sq)


__all__ = ["gated", "gated_done", "phase_mask", "settle", "upright"]
