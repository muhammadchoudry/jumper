"""Record one rollout of a jumper.cartwheel_roll checkpoint (rolling start) to
/root/scratch/roll_traj.npz for traj_video.py.

  python tools/crab_trio/roll_record.py <model_N.pt> x [task id]
"""
import sys
import torch
import numpy as np
from dataclasses import asdict
from mjrl.backend import resolve
from mjrl.backend.select import use_backend
import tasks
ckpt, out = sys.argv[1], sys.argv[2]; N = 1
res = resolve(backend="native", device="cpu", num_envs=N, cpu_threads=2, strip_visual=None, training=False)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
spec = tasks.get(sys.argv[3] if len(sys.argv) > 3 else "jumper.cartwheel_roll"); asset = spec.resolve_asset(None)
cfg = tasks.load_env_cfg(spec.id, asset=asset, play=False); cfg.scene.num_envs = N
cfg.observations["actor"].enable_corruption = False
cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
agent = tasks.load_agent_cfg(spec.id)
w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(w, asdict(agent), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
obs = w.get_observations()
robot = env.scene["robot"]
qs = []
for k in range(int(env.max_episode_length) - 2):
    with torch.no_grad():
        obs, rew, dones, extras = w.step(policy(obs))
    if dones.any(): break
    qs.append(np.concatenate([robot.data.root_link_pos_w[0].numpy() - env.scene.env_origins[0].numpy(),
                              robot.data.root_link_quat_w[0].numpy(), robot.data.joint_pos[0].numpy()]))
names = [env.scene["robot"].joint_names[i] for i in range(len(robot.data.joint_pos[0]))]
np.savez("/root/scratch/roll_traj.npz", qs=np.array(qs), names=np.array(names), dt=env.step_dt)
print("steps", len(qs))
