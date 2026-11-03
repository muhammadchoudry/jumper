"""Evaluate a jumper.cartwheel_full checkpoint from the start of the motion.

Each env starts standing (play-style spawn at the reference's beginning), trots,
jumps, rolls, and should end on its feet. Per env: how far through the timeline it
got, wheel turns while rolling (by orientation), travel, whether it ended upright,
and its final speed.

  python tools/crab_trio/full_eval.py <model_N.pt> [num_envs]     # from the repo root
"""
import math
import sys
from dataclasses import asdict

import torch
from mjrl.backend import resolve
from mjrl.backend.select import use_backend

import tasks

ckpt = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
res = resolve(backend="native", device="cpu", num_envs=N, cpu_threads=2, strip_visual=None, training=False)
use_backend(res)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl.runner import MjlabOnPolicyRunner  # noqa: E402

from tasks.jumper.jump.mdp.reference import ref_state  # noqa: E402

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
r, _, _, _ = ref_state(env, "jump")
segs = r.meta["segments_s_after_go"]
b0, b1 = segs["roll"][0], segs["roll"][1]

alive = torch.ones(N, dtype=torch.bool)
TIPS = ("LF_finger_tip_link", "RF_finger_tip_link", "LM_foot_tip_link", "RM_foot_tip_link",
        "LR_foot_tip_link", "RR_foot_tip_link")
tip_ids = [robot.body_names.index(b) for b in TIPS]
spread_sum = torch.zeros(N)
spread_n = torch.zeros(N)
last_t = torch.zeros(N)
why = [""] * N
wheel = torch.zeros(N)
prev = None
x0 = robot.data.root_link_pos_w[:, 0].clone()
x_roll0 = torch.full((N,), float("nan"))
x_roll1 = torch.full((N,), float("nan"))
for k in range(int(env.max_episode_length) - 2):
    with torch.no_grad():
        obs, rew, dones, extras = w.step(policy(obs))
    _, _, _, tsg = ref_state(env, "jump")
    q = robot.data.root_link_quat_w
    w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    bx = torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + w_ * z_), 2 * (x_ * z_ - w_ * y_)], 1)
    ang = torch.atan2(-bx[:, 2], bx[:, 0])
    rolling = alive & (tsg >= b0) & (tsg < b1)
    if prev is not None:
        dd = (ang - prev + math.pi) % (2 * math.pi) - math.pi
        wheel += torch.where(rolling, dd, torch.zeros_like(dd))
    prev = ang
    # leg spread: mean distance of the six tips from the hub, while rolling
    d = (robot.data.body_link_pos_w[:, tip_ids] - robot.data.root_link_pos_w[:, None]).norm(dim=2).mean(dim=1)
    spread_sum += torch.where(rolling, d, torch.zeros_like(d))
    spread_n += rolling.float()
    px = robot.data.root_link_pos_w[:, 0]
    x_roll0 = torch.where(rolling & torch.isnan(x_roll0), px, x_roll0)
    x_roll1 = torch.where(rolling, px, x_roll1)
    last_t = torch.where(alive, tsg, last_t)
    tm = env.termination_manager
    for i in dones.nonzero().flatten().tolist():
        if alive[i]:
            alive[i] = False
            why[i] = ",".join(n for n in tm.active_terms if tm.get_term(n)[i])
    if k == int(env.max_episode_length) - 4:
        g_end = robot.data.projected_gravity_b[:, 2].clone()
        v_end = robot.data.root_link_lin_vel_w.norm(dim=1).clone()
        z_end = robot.data.root_link_pos_w[:, 2].clone()


def stage(t):
    for name, (a, b) in segs.items():
        if t < b:
            return name
    return "end"


print(f"roll segment {b0:.2f}-{b1:.2f}s after go")
print("env: got to | wheel turns while rolling | travel while rolling | leg spread while rolling (star 0.30 m) | ends upright | final speed | why it ended")
full = 0
for i in range(N):
    up = float(g_end[i]) < -0.9
    done_ok = alive[i] or why[i] == "time_out"
    turns = abs(float(wheel[i])) / 360 * 180 / math.pi
    trav = float(x_roll1[i] - x_roll0[i]) if not math.isnan(float(x_roll0[i])) else 0.0
    if done_ok and up and turns >= 0.9:
        full += 1
    spread = float(spread_sum[i] / spread_n[i]) if spread_n[i] > 0 else float("nan")
    print(f"  {i:2d}: {stage(float(last_t[i])):10s} {turns:5.2f} turns  {trav:+5.2f} m  spread {spread:.2f} m  "
          f"{'upright' if up else 'not upright'} (g_z {float(g_end[i]):+.2f}, z {float(z_end[i]):.2f})  "
          f"{float(v_end[i]):.2f} m/s  {why[i] or ('ran to the end' if alive[i] else '')}")
print(f"full cartwheels (rolled >= 0.9 turn, ran to the end, finished upright): {full}/{N}")
