"""Film the straight-spoke cartwheel attempt (wide straight stance -> fanned crouch ->
launch -> roll over the right-hand spokes -> claw 'handstand').

  python film_wheel.py params.json out.mp4
"""
import json
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

import wheel_s as W
import wheel_s2 as W2

params, out = sys.argv[1], sys.argv[2]
W2.EDGES = np.radians([-1e3, 60, 120, 180, 240, 300])   # the edges the stage-2 result was found with
p = np.array(json.loads(open(params).read()))
states = []
r = W2.simulate(p, frame_cb=lambda t, d: states.append(d.qpos.copy()), t_end=3.6)
print({k: round(float(v), 3) for k, v in r.items()})
m, d = W._setup()
dt = m.opt.timestep
ren = mujoco.Renderer(m, 720, 1280)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[0] = opt.geomgroup[2] = 1
cam = mujoco.MjvCamera()
w = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
for t0, t1, speed, az, el, dist in [(0.0, 3.6, 1.0, 180, -10, 1.0), (1.0, 2.6, 0.25, 145, -15, 0.85)]:
    for i in range(int((t1 - t0) / speed * 30)):
        t = t0 + i * speed / 30
        d.qpos[:] = states[min(int(round(t / dt)), len(states) - 1)]
        mujoco.mj_forward(m, d)
        b = d.xpos[W.BASE]
        cam.lookat[:] = [0.5 * b[0], 0.5 * b[1], 0.12]
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        ren.update_scene(d, cam, opt)
        w.append_data(ren.render())
w.close()
print("wrote", out)
