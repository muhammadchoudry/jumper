"""Film the cartwheel: real time from the front, then 4x slow motion from a
three-quarter angle, then real time once more, closer.

The physics is simulated once and every step recorded; the cuts replay those
recorded states, so all three shots show the identical motion.

  python film_cartwheel.py out.mp4
"""
from __future__ import annotations

import json
import pathlib
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

from tools.cartwheel.physics import cartwheel as c1
from tools.cartwheel.physics import cartwheel2 as c2

HERE = pathlib.Path(__file__).resolve().parent
FPS = 30
W, H = 1280, 720


def main(out: str) -> None:
    p = np.array(json.loads((HERE.parent / "physics" / "cartwheel2_params.json").read_text()))
    states = []
    res = c2.simulate(p, frame_cb=lambda t, d: states.append(d.qpos.copy()))
    print("roll %.0f deg, upright %.2f" % (np.degrees(res["roll"]), res["up"]))
    m, d = c1._setup()
    dt = m.opt.timestep
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera()
    center = np.array([0.18, 0.13, 0.10])      # midway along the sideways travel

    shots = [  # (t_start, t_end, speed, azimuth, elevation, distance)
        (0.6, 3.4, 1.0, 180, -12, 1.05),
        (1.45, 2.75, 0.25, 140, -18, 0.85),
        (1.0, 3.2, 1.0, 200, -8, 0.8),
    ]
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                           macro_block_size=8, pixelformat="yuv420p")
    for t0, t1, speed, az, el, dist in shots:
        n = int((t1 - t0) / speed * FPS)
        for i in range(n):
            t = t0 + i * speed / FPS
            d.qpos[:] = states[min(int(round(t / dt)), len(states) - 1)]
            mujoco.mj_forward(m, d)
            body = d.xpos[c1.BASE]
            cam.lookat[:] = 0.6 * center + 0.4 * np.array([body[0], body[1], 0.1])
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            r.update_scene(d, cam, opt)
            w.append_data(r.render())
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cartwheel.mp4")
