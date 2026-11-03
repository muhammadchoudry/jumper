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
                phase_range: tuple[float, float] = (0.0, 0.95),
                roll_window: tuple[float, float] | None = None, roll_fraction: float = 0.0, **kw) -> None:
    """Reference-state spawn with a share of envs placed around the jump and landing:
    `focus_fraction` of them start between `focus_s[0]` s before the jump and
    `focus_s[1]` s after touch-down, `roll_fraction` (if `roll_window` is given) between
    those fractions of the roll segment (e.g. a figure 8's crossing), the rest
    anywhere in `phase_range`."""
    from ..cartwheel.mdp.events import reset_from_reference_phase
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    ref = get_reference(env)
    seg = ref.meta["segments_s_after_go"]
    lo = max(0.0, (float(seg["jump"][0]) + focus_s[0]) / ref.span)
    hi = min(0.99, (float(seg["roll"][0]) + focus_s[1]) / ref.span)
    u = torch.rand(len(env_ids), device=env.device)
    pick = u < focus_fraction
    if pick.any():
        reset_from_reference_phase(env, env_ids[pick], phase_range=(lo, hi), **kw)
    rest = ~pick
    if roll_window is not None and roll_fraction > 0:
        r0, r1 = float(seg["roll"][0]), float(seg["roll"][1])
        a = (r0 + roll_window[0] * (r1 - r0)) / ref.span
        b = (r0 + roll_window[1] * (r1 - r0)) / ref.span
        mid = (u >= focus_fraction) & (u < focus_fraction + roll_fraction)
        if mid.any():
            reset_from_reference_phase(env, env_ids[mid], phase_range=(a, b), **kw)
        rest = rest & ~mid
    if rest.any():
        reset_from_reference_phase(env, env_ids[rest], phase_range=phase_range, **kw)


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





# ── Steering-aware rolling (jumper.cartwheel_ten / jumper.cartwheel_eight) ─────────────
# The reference's heading may turn (a figure 8). These score the roll against the
# reference's own heading at this moment instead of world +x:
#   axle_ref  the reference body's z axis (the wheel axle), flattened to horizontal
#   travel    up x axle: the direction a wheel with that axle rolls forward


def _axle(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=1)


def _flat(a: torch.Tensor) -> torch.Tensor:
    h = a.clone()
    h[:, 2] = 0.0
    return h / h.norm(dim=1, keepdim=True).clamp(min=1e-6)


def _travel(axle_h: torch.Tensor) -> torch.Tensor:
    return torch.stack([-axle_h[:, 1], axle_h[:, 0], torch.zeros_like(axle_h[:, 0])], dim=1)


def _ref_steer(env):
    """(reference axle flattened, reference travel direction, reference yaw rate)."""
    from ..cartwheel.mdp.events import _ang_vel_table
    r, ref, _, tsg = ref_state(env, "jump")
    a = _flat(_axle(ref["base_quat"]))
    yaw_rate = _ang_vel_table(env, r)[r.index_of(tsg).round().long().clamp(0, r.n - 1), 2]
    return a, _travel(a), yaw_rate


def _lean_err(env, robot: Entity, max_lean: float = 0.4) -> torch.Tensor:
    """Lean (toward the left of travel, rad) minus what this speed and the reference's
    turn rate call for: tan(lean) = v * yaw_rate / g, as for a rolling coin."""
    _, travel_ref, yaw_rate = _ref_steer(env)
    lean = torch.asin(_axle(robot.data.root_link_quat_w)[:, 2].clamp(-1.0, 1.0))
    v = (robot.data.root_link_lin_vel_w * travel_ref).sum(dim=1).clamp(min=0.0)
    target = torch.atan(v * yaw_rate / 9.81).clamp(-max_lean, max_lean)
    return lean - target


def steer_lean(env, sigma_sq: float = 0.05, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 when the wheel leans as much as its turn needs (on edge when going straight)."""
    e = _lean_err(env, env.scene[asset_cfg.name])
    return torch.exp(-e * e / sigma_sq)


def steer_roll(env, max_rate: float = 4.0, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Rolling rate about the reference's axle (forward along its heading), clipped to
    +-max_rate, scaled to [-1, 1], times steer_lean."""
    robot: Entity = env.scene[asset_cfg.name]
    axle_ref, _, _ = _ref_steer(env)
    rate = -(robot.data.root_link_ang_vel_w * axle_ref).sum(dim=1)
    return rate.clamp(-max_rate, max_rate) / max_rate * steer_lean(env, asset_cfg=asset_cfg)


def steer_travel(env, max_speed: float = 1.2, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Hub speed along the reference's heading, clipped to +-max_speed, times steer_lean."""
    robot: Entity = env.scene[asset_cfg.name]
    _, travel_ref, _ = _ref_steer(env)
    v = (robot.data.root_link_lin_vel_w * travel_ref).sum(dim=1)
    return v.clamp(-max_speed, max_speed) / max_speed * steer_lean(env, asset_cfg=asset_cfg)


def steer_heading(env, sigma_sq: float = 0.1, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 when the wheel heads where the reference does (its axle square to that heading)."""
    robot: Entity = env.scene[asset_cfg.name]
    _, travel_ref, _ = _ref_steer(env)
    travel = _travel(_flat(_axle(robot.data.root_link_quat_w)))
    return torch.exp(-(1.0 - (travel * travel_ref).sum(dim=1)) / sigma_sq)


__all__ = ["gated", "gated_done", "gated_window", "phase_mask", "reset_focus", "settle", "steer_heading",
           "steer_lean", "steer_roll", "steer_travel", "upright"]
