"""Render a kinematic reference (the npz format of tasks/jumper/jump/mdp/reference.py:
t, base_pos, base_quat, q at rec_hz) to an mp4, side-on, plus an 8-frame sheet.

  MUJOCO_GL=egl python tools/crab_trio/ref_video.py tasks/jumper/cartwheel_jump/ref/cartwheel_jump.npz out.mp4
  (docs/media/cartwheel/cartwheel_reference4c_lean_start.mp4 is reference 4's jump npz rendered this way)
"""
import json
import sys

import imageio
import mujoco
import numpy as np

import wheel_s as W

D = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2]
hz = float(json.loads(str(D["meta"]))["rec_hz"]) if "meta" in D else 250.0
m, d = W._setup()
r = mujoco.Renderer(m, 360, 640)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[0] = opt.geomgroup[2] = 1
cam = mujoco.MjvCamera()
w = imageio.get_writer(out, fps=30, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
ts = np.arange(0, D["t"][-1], 1 / 30)
picks = set(np.linspace(0, len(ts) - 1, 8).astype(int))
frames = []
for k, tt in enumerate(ts):
    i = min(int(tt * hz), len(D["t"]) - 1)
    d.qpos[:] = 0
    d.qpos[0:3], d.qpos[3:7], d.qpos[W.QA] = D["base_pos"][i], D["base_quat"][i], D["q"][i]
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[0] * 0.8 + 0.3, 0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 90, -8
    r.update_scene(d, cam, opt)
    img = r.render()
    w.append_data(img)
    if k in picks:
        frames.append(img.copy())
w.close()
imageio.imwrite(out.replace(".mp4", "_sheet.png"), np.vstack([np.hstack(frames[:4]), np.hstack(frames[4:8])]))
print("ok", out)
