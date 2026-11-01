"""Simulate (and optionally film) the Reef Rave routine for three crabs.

  python tools/crab_trio/perform.py --check            # physics only, report balance
  python tools/crab_trio/perform.py --video out.mp4    # film it
"""
from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

import choreo
import jump
from stage import JOINTS, KD, KP, TORQUE, build, reset_home

SPACING = 0.62
CTRL_HZ = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    n = choreo.N_CRABS
    ys = [(k - (n - 1) / 2) * -SPACING for k in range(n)]  # crab 0 on the right of frame
    model = build([(0.0, y, 0.0) for y in ys], torque_control=True)
    data = mujoco.MjData(model)
    reset_home(model, data, n)

    kfs = [choreo.routine(k) for k in range(n)]
    acts = [[model.actuator(f"c{k}_{j}").id for j in JOINTS] for k in range(n)]
    lims = [model.jnt_range[model.joint(f"c0_{j}").id] for j in JOINTS]
    bases = [model.body(f"c{k}_base_link").id for k in range(n)]
    qadr = [np.array([model.jnt_qposadr[model.joint(f"c{k}_{j}").id] for j in JOINTS]) for k in range(n)]
    vadr = [np.array([model.jnt_dofadr[model.joint(f"c{k}_{j}").id] for j in JOINTS]) for k in range(n)]
    lo = np.array([lim[0] for lim in lims])
    hi = np.array([lim[1] for lim in lims])
    jump_starts = [choreo.jumps(k) for k in range(n)]
    dance_tgt = [None] * n

    dt = model.opt.timestep
    steps_per_ctrl = round(1 / CTRL_HZ / dt)
    steps_per_frame = round(1 / args.fps / dt)
    total = int(choreo.END_BEAT * choreo.BEAT / dt)

    renderer = writer = None
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, args.height, args.width)
        # label the file with the true frame rate so video time == sim time
        writer = imageio.get_writer(args.video, fps=1 / (steps_per_frame * dt), codec="libx264", quality=8,
                                    macro_block_size=8, pixelformat="yuv420p")
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = opt.geomgroup[2] = 1
        cam = mujoco.MjvCamera()

    worst = [1.0] * n
    log = []
    for step in range(total):
        t = step * dt
        if step % steps_per_ctrl == 0:
            for k in range(n):
                tgt = choreo.targets(kfs[k], t)
                dance_tgt[k] = np.clip([tgt[j] for j in JOINTS], lo, hi)
        for k in range(n):
            q, qd = data.qpos[qadr[k]], data.qvel[vadr[k]]
            in_jump = [s0 for s0 in jump_starts[k] if s0 <= t < s0 + jump.DURATION]
            if in_jump:  # the jump's own servo commands, gains and motor envelope
                c = jump.command(t - in_jump[0])
                tau = jump.torque(np.array([c[j] for j in JOINTS]), q, qd)
            else:        # dance servos: kp 10, kd 0.5, 1.75 N*m
                tau = np.clip(KP * (dance_tgt[k] - q) - KD * qd, -TORQUE, TORQUE)
            data.ctrl[acts[k]] = tau
        mujoco.mj_step(model, data)
        ups = [data.xmat[b][8] for b in bases]
        worst = [min(w, u) for w, u in zip(worst, ups)]
        if step % int(choreo.BEAT / dt) == 0:
            log.append((t / choreo.BEAT, ups, [data.xpos[b][2] for b in bases]))
        if writer and step % steps_per_frame == 0:
            # slow drift from a front-left three-quarter view to front-right
            u = t / (total * dt)
            cam.lookat[:] = [0.0, 0.0, 0.09]
            cam.distance = 1.62 - 0.08 * math.sin(math.pi * u)
            cam.azimuth = 160 + 40 * u
            cam.elevation = -24
            renderer.update_scene(data, cam, opt)
            writer.append_data(renderer.render())

    if writer:
        writer.close()
    for beat, ups, zs in log:
        flag = " <-- tilted" if min(ups) < 0.9 else ""
        print(f"beat {beat:5.1f}  up " + " ".join(f"{u:.2f}" for u in ups)
              + "  z " + " ".join(f"{z:.3f}" for z in zs) + flag)
    print("worst uprightness per crab:", [round(w, 3) for w in worst])
    fallen = any(w < 0.5 for w in worst)
    print("FELL" if fallen else "all three stayed on their feet")


if __name__ == "__main__":
    main()
