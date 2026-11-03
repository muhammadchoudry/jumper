"""Record jumper.cartwheel_full rollouts from a standing start and keep the best one
(an env that ran to the end and finished on its feet; otherwise the one that survived longest) as an npz for
traj_video.py.

  python tools/crab_trio/full_record.py <model_N.pt> out.npz [num_envs] [seed]
"""
import sys
from dataclasses import asdict

import numpy as np
import torch
from mjrl.backend import resolve
from mjrl.backend.select import use_backend

import tasks

ckpt, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
torch.manual_seed(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
res = resolve(backend="native", device="cpu", num_envs=N, cpu_threads=2, strip_visual=None, training=False)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl.runner import MjlabOnPolicyRunner  # noqa: E402

spec = tasks.get("jumper.cartwheel_full")
cfg = tasks.load_env_cfg(spec.id, asset=spec.resolve_asset(None), play=False)
cfg.scene.num_envs = N
cfg.observations["actor"].enable_corruption = False
cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.001)
if "focus_fraction" in cfg.events["reset_from_reference"].params:
    cfg.events["reset_from_reference"].params["focus_fraction"] = 0.0
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
agent = tasks.load_agent_cfg(spec.id)
w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(w, asdict(agent), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
obs = w.get_observations()
robot = env.scene["robot"]
frames, alive, steps = [], torch.ones(N, dtype=torch.bool), torch.zeros(N)
for k in range(int(env.max_episode_length) - 2):
    with torch.no_grad():
        obs, rew, dones, extras = w.step(policy(obs))
    alive &= ~dones.bool()
    steps += alive.float()
    frames.append(np.concatenate([(robot.data.root_link_pos_w - env.scene.env_origins).numpy(),
                                  robot.data.root_link_quat_w.numpy(), robot.data.joint_pos.numpy()], 1))
    if not alive.any():
        break
upright = (robot.data.projected_gravity_b[:, 2] < -0.9).float()   # ended on its feet
best = int(torch.argmax(steps + 1000.0 * upright * alive.float() + 1e-3 * robot.data.root_link_pos_w[:, 0]))
qs = np.array(frames)[: int(steps[best]) + 1, best]
np.savez(out, qs=qs, names=np.array(robot.joint_names), dt=env.step_dt)
print(f"best env {best}: {int(steps[best])} steps ({steps[best] * env.step_dt:.2f} s); survivors {int(alive.sum())}/{N}")
