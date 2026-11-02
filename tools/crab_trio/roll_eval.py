"""Evaluate a jumper.cartwheel_roll checkpoint from a rolling start on edge.

Per env: time on edge, wheel turned by orientation vs integrated body spin (they
disagree when it spins like a coin instead of rolling), forward travel, worst lean.

  python tools/crab_trio/roll_eval.py <model_N.pt>        # from the repo root
"""
import sys
import math
import torch
from dataclasses import asdict
from mjrl.backend import resolve
from mjrl.backend.select import use_backend
import tasks
ckpt = sys.argv[1]; N = 16
res = resolve(backend="native", device="cpu", num_envs=N, cpu_threads=2, strip_visual=None, training=False)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
spec = tasks.get("jumper.cartwheel_roll"); asset = spec.resolve_asset(None)
cfg = tasks.load_env_cfg(spec.id, asset=asset, play=False); cfg.scene.num_envs = N
cfg.observations["actor"].enable_corruption = False
cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)   # all start at the roll start
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
agent = tasks.load_agent_cfg(spec.id)
w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(w, asdict(agent), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
obs = w.get_observations()
robot = env.scene["robot"]
x0 = robot.data.root_link_pos_w[:, 0].clone()
wheel = torch.zeros(N); alive = torch.ones(N, dtype=torch.bool); prev = None
spin_int = torch.zeros(N); tilt_max = torch.zeros(N); t_alive = torch.zeros(N); xs = torch.zeros(N)
traj = []
for k in range(int(env.max_episode_length) - 2):
    with torch.no_grad():
        obs, rew, dones, extras = w.step(policy(obs))
    q = robot.data.root_link_quat_w; w_, x_, y_, z_ = q[:,0], q[:,1], q[:,2], q[:,3]
    bx = torch.stack([1-2*(y_*y_+z_*z_), 2*(x_*y_+w_*z_), 2*(x_*z_-w_*y_)],1)
    ang = torch.atan2(-bx[:,2], bx[:,0])
    if prev is not None:
        dd = (ang - prev + math.pi) % (2*math.pi) - math.pi
        wheel += torch.where(alive, dd, torch.zeros_like(dd))
    prev = ang
    spin_int += torch.where(alive, -robot.data.root_link_ang_vel_b[:, 2] * env.step_dt, torch.zeros(N))
    gz = robot.data.projected_gravity_b[:, 2].abs()
    tilt_max = torch.where(alive, torch.maximum(tilt_max, torch.rad2deg(torch.asin(gz.clamp(max=1)))), tilt_max)
    t_alive += alive.float() * env.step_dt
    xs = torch.where(alive, robot.data.root_link_pos_w[:, 0] - x0, xs)
    if k % 10 == 0:
        traj.append((k*env.step_dt, float(robot.data.root_link_pos_w[0,0]-x0[0]), float(robot.data.root_link_pos_w[0,2]), math.degrees(float(wheel[0])), float(-robot.data.root_link_ang_vel_b[0,2]), float(robot.data.root_link_lin_vel_w[0,0])))
    alive &= ~dones.bool()
print("per env: time on edge (s), wheel turned about its axle by orientation (deg), integrated spin about axle (deg), forward travel (m), worst lean (deg)")
for i in range(N):
    print(f"  {i:2d}: {t_alive[i]:.2f}s  wheel {math.degrees(float(wheel[i])):7.0f}  spin {math.degrees(float(spin_int[i])):7.0f}  travel {float(xs[i]):+.2f} m  lean {float(tilt_max[i]):3.0f}")
print("env 0 trace (t, x, hub z, wheel deg, spin rad/s, vx):")
for t in traj[::5]: print("  ", tuple(round(v, 2) for v in t))
