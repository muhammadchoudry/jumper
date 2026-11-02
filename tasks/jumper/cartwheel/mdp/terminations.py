"""Terminations for the cartwheel.

The jump's `bad_orientation` (tilt > 1.2 rad) cannot be used: a cartwheel goes
upside down on purpose. Instead an episode ends when the body's orientation
strays from the reference's -- measured, like the reward, on the gravity
direction in the base frame (what the IMU sees).
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...jump.mdp.reference import ref_state


def orientation_diverged(
    env,
    command_name: str = "jump",
    max_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the angle between the robot's and the reference's gravity
    directions (base frame) exceeds `max_angle` radians."""
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    cos = (robot.data.projected_gravity_b * ref["proj_g"]).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos) > max_angle




def _quat_error_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Smallest rotation angle (rad) between two batches of unit quaternions (w, x, y, z)."""
    dot = (q1 * q2).sum(dim=1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def rotation_diverged(
    env,
    command_name: str = "jump",
    max_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the body's full orientation strays from the reference's.

    `orientation_diverged` compares only the gravity direction, which cannot see
    a rotation about the vertical: a policy lying ~70 degrees off its edge passed
    it for the whole roll while the reference turned 360 degrees (seen in
    jumper.cartwheel_runup, round 1700). The full rotation error catches that.
    The heading is part of the reference (it runs straight along +x).
    """
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    return _quat_error_angle(robot.data.root_link_quat_w, ref["base_quat"]) > max_angle


__all__ = ["orientation_diverged", "rotation_diverged"]
