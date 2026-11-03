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


def _seg_time(env, name: str | None, end: bool = False) -> float:
    if name is None:
        return float("inf") if end else float("-inf")
    seg = get_reference(env).meta["segments_s_after_go"]
    return float(seg[name][0])


def gated_window(env, fn, start: str | None = None, end: str | None = None, **params) -> torch.Tensor:
    """A termination `fn` that only ends episodes between the start of segment
    `start` and the start of segment `end` (None = open), e.g. the jump itself:
    start="jump", end="roll" stops at touch-down."""
    _, _, _, tsg = ref_state(env, "jump")
    m = (tsg >= _seg_time(env, start)) & (tsg < _seg_time(env, end, end=True))
    return fn(env, **params) & m


def reset_focus(env, env_ids, focus_s: tuple[float, float] = (-0.3, 0.4), focus_fraction: float = 0.35,
                phase_range: tuple[float, float] = (0.0, 0.95), **kw) -> None:
    """Reference-state spawn with a share of envs placed around the jump and landing:
    `focus_fraction` of them start between `focus_s[0]` s before the jump and
    `focus_s[1]` s after touch-down, the rest anywhere in `phase_range`."""
    from ..cartwheel.mdp.events import reset_from_reference_phase
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    ref = get_reference(env)
    seg = ref.meta["segments_s_after_go"]
    lo = max(0.0, (float(seg["jump"][0]) + focus_s[0]) / ref.span)
    hi = min(0.99, (float(seg["roll"][0]) + focus_s[1]) / ref.span)
    pick = torch.rand(len(env_ids), device=env.device) < focus_fraction
    if pick.any():
        reset_from_reference_phase(env, env_ids[pick], phase_range=(lo, hi), **kw)
    if (~pick).any():
        reset_from_reference_phase(env, env_ids[~pick], phase_range=phase_range, **kw)


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


__all__ = ["gated", "gated_done", "gated_window", "phase_mask", "reset_focus", "settle", "upright"]
