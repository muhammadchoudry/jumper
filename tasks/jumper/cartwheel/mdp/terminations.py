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


__all__ = ["orientation_diverged"]
