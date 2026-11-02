"""Cartwheel-specific reward: track the full body orientation.

The jump's `ref_base_ori` scores the gravity direction only (what an IMU sees),
which is blind to the wheel's rotation about the vertical. This term scores the
whole rotation, so actually rolling with the reference pays.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...jump.mdp.reference import ref_state
from .terminations import _quat_error_angle


def ref_base_rotation(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = _quat_error_angle(robot.data.root_link_quat_w, ref["base_quat"])
    return torch.exp(-(e * e) / sigma_sq)


__all__ = ["ref_base_rotation"]
