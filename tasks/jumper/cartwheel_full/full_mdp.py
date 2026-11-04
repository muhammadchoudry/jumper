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





# ── Figure 8 by distance rolled (jumper.cartwheel_eight, try 3) ────────────────────────
# Tries 1-2 steered toward the reference's heading at each moment of the reference's
# clock: once the robot was ahead of or behind the reference, the turn targets fell in
# the wrong places, and it never learned the right-hand loop. Here each env draws its
# own figure 8 from where it lands: the target heading is a function of the distance
# it has rolled since the roll began (EIGHT_LENGTH m for the whole 8), measured from
# its own heading at that moment, and mirrored (right loop first) in half the envs.

EIGHT_LENGTH = 8.0          # m of path for the whole figure 8 (~1.1 m/s for 7.5 s)
_EIGHT_PROFILE = None


def _eight_profile_t(device):
    """(s grid, heading, curvature per unit s) for the crossing-diagonals 8."""
    global _EIGHT_PROFILE
    if _EIGHT_PROFILE is None or _EIGHT_PROFILE[0].device != torch.device(device):
        import importlib.util
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[3] / "tools/crab_trio/cartwheel_ref5.py"
        spec = importlib.util.spec_from_file_location("_cartwheel_ref5", p)
        # cartwheel_ref5 imports its siblings by name: put tools/crab_trio on the path for the load
        import sys
        sys.path.insert(0, str(p.parent))
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            s, psi = mod._eight_profile()
        finally:
            sys.path.remove(str(p.parent))
        s = torch.as_tensor(s, dtype=torch.float32, device=device)
        psi = torch.as_tensor(psi, dtype=torch.float32, device=device)
        kappa = torch.gradient(psi, spacing=(s,))[0]
        _EIGHT_PROFILE = (s, psi, kappa)
    return _EIGHT_PROFILE


def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    i = torch.searchsorted(xs, x.clamp(xs[0], xs[-1])).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    w = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0, 1)
    return y0 + w * (y1 - y0)


def _heading(q: torch.Tensor) -> torch.Tensor:
    """Travel heading of a wheel with this orientation (up x axle), rad."""
    t = _travel(_flat(_axle(q)))
    return torch.atan2(t[:, 1], t[:, 0])


def eight_state(env, asset_cfg: SceneEntityCfg = _ROBOT):
    """Per env, this step: (target heading, target curvature per m, distance rolled,
    mirror sign, in-roll mask). Cached per step; state lives on the env."""
    stamp = int(env.common_step_counter)
    if getattr(env, "_eight_stamp", None) == stamp:
        return env._eight_cache
    robot: Entity = env.scene[asset_cfg.name]
    n, dev = env.num_envs, env.device
    if not hasattr(env, "_eight_d"):
        env._eight_d = torch.zeros(n, device=dev)
        env._eight_psi0 = torch.full((n,), float("nan"), device=dev)
        env._eight_sign = torch.ones(n, device=dev)
        env._eight_t = torch.zeros(n, device=dev)
    _, _, _, tsg = ref_state(env, "jump")
    in_roll = phase_mask(env, "B")
    # a new episode: fresh distance and heading, a random direction; envs born mid-roll
    # start somewhere along their 8, so every part of it (the crossing too) gets practice
    new = env.episode_length_buf <= 1
    if new.any():
        env._eight_psi0[new] = float("nan")
        env._eight_sign[new] = torch.where(torch.rand(int(new.sum()), device=dev) < 0.5, -1.0, 1.0)
        born_in_roll = new & in_roll
        env._eight_d[new] = 0.0
        env._eight_d[born_in_roll] = torch.rand(int(born_in_roll.sum()), device=dev) * 0.9 * EIGHT_LENGTH
    heading = _heading(robot.data.root_link_quat_w)
    s_grid, psi_grid, kappa_grid = _eight_profile_t(dev)
    frac = (env._eight_d / EIGHT_LENGTH).clamp(0, 1)
    prof = _interp(frac, s_grid, psi_grid)
    start = in_roll & torch.isnan(env._eight_psi0)
    if start.any():                         # psi0 is where the 8 would have started from
        env._eight_psi0[start] = heading[start] - env._eight_sign[start] * prof[start]
    v = (robot.data.root_link_lin_vel_w[:, :2] * torch.stack([torch.cos(heading), torch.sin(heading)], 1)).sum(1)
    env._eight_d = torch.where(in_roll, env._eight_d + v.clamp(min=0) * env.step_dt, env._eight_d)
    frac = (env._eight_d / EIGHT_LENGTH).clamp(0, 1)
    target = env._eight_psi0 + env._eight_sign * _interp(frac, s_grid, psi_grid)
    done = env._eight_d >= EIGHT_LENGTH
    kappa = torch.where(done, torch.zeros_like(frac),
                        env._eight_sign * _interp(frac, s_grid, kappa_grid) / EIGHT_LENGTH)
    env._eight_cache = (torch.nan_to_num(target), kappa, env._eight_d, env._eight_sign, in_roll)
    env._eight_stamp = stamp
    return env._eight_cache


def eight_lean(env, sigma_sq: float = 0.05, max_lean: float = 0.4, asset_cfg: SceneEntityCfg = _ROBOT):
    """1 when the lean (toward the left of travel) is what the 8's curvature here needs
    at this speed: tan(lean) = v^2 * curvature / g."""
    robot: Entity = env.scene[asset_cfg.name]
    _, kappa, _, _, _ = eight_state(env, asset_cfg)
    lean = torch.asin(_axle(robot.data.root_link_quat_w)[:, 2].clamp(-1.0, 1.0))
    heading = _heading(robot.data.root_link_quat_w)
    v = (robot.data.root_link_lin_vel_w[:, :2] * torch.stack([torch.cos(heading), torch.sin(heading)], 1)).sum(1)
    target = torch.atan(v.clamp(min=0) ** 2 * kappa / 9.81).clamp(-max_lean, max_lean)
    e = lean - target
    return torch.exp(-e * e / sigma_sq)


def eight_heading(env, sigma_sq: float = 0.1, asset_cfg: SceneEntityCfg = _ROBOT):
    """1 when the travel heading matches the 8's at the distance rolled."""
    robot: Entity = env.scene[asset_cfg.name]
    target, _, _, _, _ = eight_state(env, asset_cfg)
    err = _heading(robot.data.root_link_quat_w) - target
    return torch.exp(-(1.0 - torch.cos(err)) / sigma_sq)


def eight_roll(env, max_rate: float = 4.0, asset_cfg: SceneEntityCfg = _ROBOT):
    """Rolling rate about the robot's own (horizontal) axle, scaled to [-1, 1], times
    eight_lean. A coin spinning in place turns about the vertical: it scores nothing."""
    robot: Entity = env.scene[asset_cfg.name]
    axle = _flat(_axle(robot.data.root_link_quat_w))
    rate = -(robot.data.root_link_ang_vel_w * axle).sum(dim=1)
    return rate.clamp(-max_rate, max_rate) / max_rate * eight_lean(env, asset_cfg=asset_cfg)


def eight_travel(env, max_speed: float = 1.2, asset_cfg: SceneEntityCfg = _ROBOT):
    """Hub speed along its own heading, scaled to [-1, 1], times eight_heading (so
    rolling counts when it goes the 8's way)."""
    robot: Entity = env.scene[asset_cfg.name]
    heading = _heading(robot.data.root_link_quat_w)
    v = (robot.data.root_link_lin_vel_w[:, :2] * torch.stack([torch.cos(heading), torch.sin(heading)], 1)).sum(1)
    return v.clamp(-max_speed, max_speed) / max_speed * eight_heading(env, asset_cfg=asset_cfg)


__all__ = ["eight_heading", "eight_lean", "eight_roll", "eight_state", "eight_travel", "gated", "gated_done",
           "gated_window", "phase_mask", "reset_focus", "settle", "steer_heading", "steer_lean", "steer_roll",
           "steer_travel", "upright"]
