"""The crab as a rigid six-spoke star cartwheeling down a slope -- physics only.

On flat ground a rigid star stops within a turn (~2/3 of its spin energy goes at
each spoke change). Downhill, every spoke change also drops the hub by
step * sin(slope), and above a certain slope that pays for the loss: the star
settles into a steady roll. Here the legs are just held in the flat star by the
joint PD (kp 20, kd 0.5, 1.75 N m), no learning, no help.

The slope is simulated by tilting gravity (identical physics); `render` turns the
result back into a scene with a real tilted hill for the video.

  python hill_roll.py sweep                 # slope x start spin: turns, lean, how it ended
  python hill_roll.py run SLOPE_DEG W0 OUT.npz [T_END]
"""
from __future__ import annotations

import math
import sys

import mujoco
import numpy as np

import jump
import wheel_s as W
from limb_wheel import ANG, HIP, KIN, LEGV, STAR, start_state, wrap180

#: balance: legs near the ground (rest angle within [win_lo, win_hi] deg ahead of
#: straight down) are tilted out of the wheel's plane, straight, toward the side the
#: wheel leans -- it steps under itself, as a rolling coin steers into its lean
BAL = dict(k=1.0, kd=0.15, win_lo=-40.0, win_hi=80.0)


def tilt_targets(lean, lean_rate, theta, p):
    """Foot targets in the body frame: each straight star leg, tilted about its hip."""
    out = np.zeros((6, 3))
    for i in range(6):
        v = LEGV[i]
        L, r = np.linalg.norm(v), math.hypot(v[0], v[1])
        phi = 0.0 if p.get("flat") else math.atan2(v[2], r)
        x = wrap180(math.degrees(ANG[i]) + 90.0 - math.degrees(theta))
        if p["win_lo"] <= x <= p["win_hi"]:
            phi += -p["k"] * (lean + p["kd"] * lean_rate)
        phi = max(-0.9, min(0.9, phi))
        out[i] = HIP[i] + L * np.array([v[0] / r * math.cos(phi), v[1] / r * math.cos(phi), math.sin(phi)])
    return out
from stage import JOINTS

G = 9.81


def simulate(slope_deg: float, w0: float, t_end: float = 10.0, record: bool = False, flat_after: float | None = None,
             bal: dict | None = None):
    """Roll the rigid star down a slope of `slope_deg` (down-slope is +x in the slope
    frame). If `flat_after` (m of travel) is set, the slope levels out there (gravity
    turns back to straight down) -- a run-out at the bottom of the hill."""
    m, d = W._setup()
    s = math.radians(slope_deg)
    start_state(m, d, w0)
    dt = m.opt.timestep
    theta, prev = 0.0, None
    qs, lean_max, why = [], 0.0, "time"
    x0 = float(d.xpos[W.BASE][0])
    tgt, lean_prev = STAR.copy(), 0.0
    for step in range(int(t_end / dt)):
        flat = flat_after is not None and d.xpos[W.BASE][0] - x0 > flat_after
        m.opt.gravity[:] = [0, 0, -G] if flat else [G * math.sin(s), 0, -G * math.cos(s)]
        R = d.xmat[W.BASE].reshape(3, 3)
        bx, axle = R[:, 0], R[:, 2]
        ang = math.atan2(-bx[2], bx[0])
        if prev is not None:
            theta += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        lean = math.asin(max(-1.0, min(1.0, axle[2])))
        lean_max = max(lean_max, abs(lean))
        lean_rate, lean_prev = (lean - lean_prev) / dt, lean
        if bal is not None and step % 5 == 0:
            tgt = KIN.ik(tgt, tilt_targets(lean, lean_rate, theta, bal))
        d.ctrl[W.AI] = jump.torque(tgt, d.qpos[W.QA], d.qvel[W.VA])
        mujoco.mj_step(m, d)
        if record and step % 5 == 0:
            qs.append(np.concatenate([d.qpos[0:7], d.qpos[W.QA]]))
        if abs(lean) > 1.2:
            why = "fell over"
            break
        if d.xpos[W.BASE][2] < 0.12:
            why = "hub down"
            break
    m.opt.gravity[:] = [0, 0, -G]
    out = dict(turns=abs(theta) / (2 * math.pi), t=step * dt, travel=float(d.xpos[W.BASE][0]) - x0,
               side=float(d.xpos[W.BASE][1]), lean_max=math.degrees(lean_max), spin=float(-d.qvel[5]), why=why)
    if record:
        out["qs"] = np.array(qs)
        out["dt"] = dt * 5
    return out


def sweep():
    for slope in (4, 6, 8, 10, 12, 15):
        for w0 in (2.4, 4.0):
            r = simulate(slope, w0, t_end=8.0)
            print(f"slope {slope:4.1f} deg  w0 {w0:.1f}: {r['turns']:5.2f} turns in {r['t']:4.1f} s, "
                  f"{r['travel']:5.2f} m down, {r['side']:+.2f} m sideways, max lean {r['lean_max']:4.0f} deg, "
                  f"spin at end {r['spin']:5.2f} rad/s  ({r['why']})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep()
    elif sys.argv[1] == "run":
        slope, w0, out = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        t_end = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
        r = simulate(slope, w0, t_end, record=True)
        np.savez(out, qs=r.pop("qs"), dt=r.pop("dt"), names=np.array(JOINTS), slope=slope)
        print(r)
