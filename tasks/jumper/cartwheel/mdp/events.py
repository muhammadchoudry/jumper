"""Reference-state reset that keeps the spin.

`jumper.jump`'s reset zeroes angular velocity, which is harmless for a jump
(the body tilts < 2 degrees) and fatal for a cartwheel: an env born upside down
mid-roll with no spin just falls. Here the spawn takes the reference's roll rate
(`base_ang_vel`, world frame) as well.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...jump.mdp.reference import get_reference

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _ang_vel_table(env, ref) -> torch.Tensor:
    if not hasattr(env, "_cartwheel_ang_vel"):
        d = np.load(ref.path, allow_pickle=True)
        env._cartwheel_ang_vel = torch.as_tensor(d["base_ang_vel"], dtype=torch.float32,
                                                 device=env.device)
    return env._cartwheel_ang_vel


def reset_from_reference_phase(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "jump",
    rsi_fraction: float = 1.0,
    phase_range: tuple[float, float] = (0.0, 0.9),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    del command_name, asset_cfg
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    robot: Entity = env.scene["robot"]
    ref = get_reference(env)

    r = torch.empty(n, device=env.device)
    is_rsi = r.uniform_(0.0, 1.0) < rsi_fraction
    phase = r.uniform_(*phase_range)
    if not hasattr(env, "_jump_rsi_phase"):
        env._jump_rsi_phase = torch.zeros(env.num_envs, device=env.device)
    env._jump_rsi_phase[env_ids] = torch.where(is_rsi, phase, torch.zeros_like(phase))

    rsi_ids = env_ids[is_rsi]
    if len(rsi_ids) == 0:
        return
    rsi_local = is_rsi.nonzero(as_tuple=False).flatten()
    tsg = phase[rsi_local] * ref.span
    idx = ref.index_of(tsg)
    s = ref.sample(idx)
    ang = _ang_vel_table(env, ref)[idx.round().long().clamp(0, ref.n - 1)]

    pos = env.scene.env_origins[rsi_ids] + s["base_pos"]
    root_state = torch.cat([pos, s["base_quat"], s["base_lin_vel"], ang], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=rsi_ids)
    robot.write_joint_state_to_sim(s["q"], s["qd"], env_ids=rsi_ids)


__all__ = ["reset_from_reference_phase"]
