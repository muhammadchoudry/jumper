"""Record a physics run-up for reference 3: the repo's trained walking policy
(`jumper.posture`, tasks/jumper/posture/out/example/model_74800.pt, tripod gait)
trotting straight forward at a fixed velocity command, from a standing start.

Writes tools/crab_trio/runup_walk.npz with time, base pose/velocity and joint
angles at the walking task's control rate.

  python record_runup.py [vx=0.8] [seconds=2.0]
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(vx: float = 0.8, seconds: float = 2.0) -> None:
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend="native", device="cpu", num_envs=1, cpu_threads=1, strip_visual=None,
                  training=False)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    spec = tasks.get("jumper.posture")
    asset = spec.resolve_asset(None)
    ckpt = ROOT / "tasks/jumper/posture/out/example/model_74800.pt"
    cfg = tasks.load_env_cfg(spec.id, asset=asset)          # training config, commands pinned:
    tw = cfg.commands["twist"]                                # straight ahead at vx, no turning
    tw.ranges.lin_vel_x = (vx, vx)
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.ranges.heading = None
    for k in ("rel_standing_envs", "rel_heading_envs"):
        if hasattr(tw, k):
            setattr(tw, k, 0.0)
    if hasattr(tw, "heading_command"):
        tw.heading_command = False
    po = cfg.commands["posture"].ranges                       # neutral body posture
    po.twist = po.pitch = po.roll = 0.0
    po.height = (0.1065, 0.1065)
    cfg.curriculum = {}
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    agent = tasks.load_agent_cfg(spec.id)
    w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(w, asdict(agent), device="cpu")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    robot = env.scene["robot"]
    obs = w.get_observations()
    rows = dict(t=[], base_pos=[], base_quat=[], base_lin_vel=[], base_ang_vel_w=[], q=[])
    origin = env.scene.env_origins[0]
    for k in range(int(seconds / env.step_dt)):
        d = robot.data
        rows["t"].append(k * env.step_dt)
        rows["base_pos"].append((d.root_link_pos_w[0] - origin).tolist())
        rows["base_quat"].append(d.root_link_quat_w[0].tolist())
        rows["base_lin_vel"].append(d.root_link_lin_vel_w[0].tolist())
        rows["base_ang_vel_w"].append(d.root_link_ang_vel_w[0].tolist())
        rows["q"].append(d.joint_pos[0].tolist())
        with torch.no_grad():
            obs, _, dones, _ = w.step(policy(obs))
        if bool(dones[0]):
            raise SystemExit("walking episode ended early -- the run-up fell")
    out = pathlib.Path(__file__).with_name("runup_walk.npz")
    np.savez(out, joint_names=np.array(robot.joint_names), dt=env.step_dt, vx=vx,
             **{k: np.array(v) for k, v in rows.items()})
    v = np.array(rows["base_lin_vel"])
    print(f"wrote {out}: {len(rows['t'])} steps at {1 / env.step_dt:.0f} Hz; "
          f"forward speed over the last second {v[-int(1 / env.step_dt):, 0].mean():.2f} m/s, "
          f"travelled {rows['base_pos'][-1][0] - rows['base_pos'][0][0]:.2f} m")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.8, float(a[1]) if len(a) > 1 else 2.0)
