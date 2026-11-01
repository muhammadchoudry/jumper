"""Film the claw handstand work.

  python film_handstand.py out.mp4 [entry_params.json]

Shot 1: balanced claw handstand + 3 push-ups (wrist reflex), side and 3/4 views.
        The handstand is *placed* at the start (labelled so in the report) --
        it shows the hold and the push-ups, not how to get up there.
Shot 2 (if entry params given): the pike-press entry attempt from standing.
"""
from __future__ import annotations

import json
import math
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

import handstand as H

sys.path.insert(0, "/root/scratch")


def record_pushups():
    import handstand_pushup
    states = []
    handstand_pushup.run(1.3, 0, 4.5, reps=3, record=lambda t, d: states.append(d.qpos.copy()))
    return states


def record_entry(p):
    states = []
    r = H.pike_sim(np.asarray(p), record=lambda t, d: states.append(d.qpos.copy()))
    print("entry:", r)
    return states


def render(states, w, views, speed=1.0, t0=0.0, t1=None):
    m, d = H.setup()
    dt = m.opt.timestep
    ren = mujoco.Renderer(m, 720, 1280)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera()
    t1 = t1 if t1 is not None else len(states) * dt
    for az, el, dist in views:
        for i in range(int((t1 - t0) / speed * 30)):
            t = t0 + i * speed / 30
            d.qpos[:] = states[min(int(round(t / dt)), len(states) - 1)]
            mujoco.mj_forward(m, d)
            b = d.xpos[H.BASE]
            cam.lookat[:] = [b[0], b[1], 0.16]
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            ren.update_scene(d, cam, opt)
            w.append_data(ren.render())


if __name__ == "__main__":
    out = sys.argv[1]
    w = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p")
    if len(sys.argv) > 2:
        st = record_entry(json.loads(open(sys.argv[2]).read()))
        render(st, w, [(90, -8, 1.0)], t0=0.0, t1=3.4)
        render(st, w, [(120, -10, 0.9)], speed=0.25, t0=2.1, t1=2.7)   # 4x slow motion
    st = record_pushups()
    render(st, w, [(90, -5, 0.95)], t0=0.3, t1=None)
    render(st, w, [(140, -15, 0.85)], t0=0.3, t1=3.5)
    w.close()
    print("wrote", out)
    del math
