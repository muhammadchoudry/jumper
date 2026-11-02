"""Rewards and termination for jumper.cartwheel_roll.

Frames: the star's wheel axle is the body z axis. On its right edge rolling
forward (+x), body z points along world -y and the wheel turns about body -z.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_ROBOT = SceneEntityCfg("robot")


def _axle_w(robot: Entity) -> torch.Tensor:
    """Body z axis in the world frame (the wheel axle), from the base quaternion (w, x, y, z)."""
    q = robot.data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=1)


def on_edge(env, sigma_sq: float = 0.05, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 when the wheel stands on its edge (axle horizontal), falling off as it leans."""
    robot: Entity = env.scene[asset_cfg.name]
    gz = robot.data.projected_gravity_b[:, 2]
    return torch.exp(-gz * gz / sigma_sq)


def roll_forward(env, max_rate: float = 4.0, sigma_sq: float = 0.05,
                 asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Wheel spin (rad/s about body -z), clipped to [-max_rate, max_rate] and scaled to
    [-1, 1], counted only while on edge -- so spinning flat on its back pays nothing,
    and signed, so rocking back and forth nets nothing."""
    robot: Entity = env.scene[asset_cfg.name]
    rate = (-robot.data.root_link_ang_vel_b[:, 2]).clamp(-max_rate, max_rate) / max_rate
    return rate * on_edge(env, sigma_sq, asset_cfg)


def travel_forward(env, max_speed: float = 1.2, sigma_sq: float = 0.05,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Speed of the hub along +x (m/s), clipped to +-max_speed, signed, while on edge."""
    robot: Entity = env.scene[asset_cfg.name]
    vx = robot.data.root_link_lin_vel_w[:, 0].clamp(-max_speed, max_speed) / max_speed
    return vx * on_edge(env, sigma_sq, asset_cfg)


def straight(env, sigma_sq: float = 0.05, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """1 while the axle stays square to the direction of travel (no x component)."""
    robot: Entity = env.scene[asset_cfg.name]
    ax = _axle_w(robot)[:, 0]
    return torch.exp(-ax * ax / sigma_sq)


def fell_off_edge(env, max_tilt_cos: float = 0.7, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """True once the wheel has toppled: |gravity along the axle| > max_tilt_cos (~44 deg)."""
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.projected_gravity_b[:, 2].abs() > max_tilt_cos


__all__ = ["fell_off_edge", "on_edge", "roll_forward", "straight", "travel_forward"]
