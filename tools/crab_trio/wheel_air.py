"""Aerial cartwheel: a lopsided version of the robot's own jump, a mid-air switch
into the straight-spoke star, then roll out on the spokes and land.

Launch: the repo's high-jump servo commands (tasks/jumper/jump/ref), but each
group of limbs (left legs, right legs, claws) gets its own amplitude scale and
time shift, so one side drives harder or earlier and the body spins sideways as
it rises. In the air the limbs carry no load, so they can snap into the star
(~0.1 s) within the motor limits. Then the roll-scheduled straight-spoke
controller and balance reflex from wheel_s2.py take over.

  python wheel_air.py launch     # stage 1: search the lopsided jump + switch time
  python wheel_air.py spokes     # stage 2: keep the launch, search the spokes
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

import jump
import wheel_s as W
import wheel_s2 as W2
from stage import JOINTS

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "wheel_air_params.json"
LAUNCH = HERE / "wheel_air_launch.json"

GROUP = {j: ("C" if j[1] == "F" else j[0]) for j in JOINTS}   # C = claws, L / R = legs
STANDV = np.array([jump.STAND[j] for j in JOINTS])
QCMD = np.array([[dict(zip(jump.ORDER, row))[j] for j in JOINTS] for row in jump.Q_CMD])
T_READY = 0.8                    # ease into the jump's stand pose

# params: [sL, sR, sC, dL, dR, dC, t_switch] + sector offsets (36) + balance (2)
I_L, I_SECT = 0, 7
I_BAL = I_SECT + 6 * W.N_SECT
NP = I_BAL + 2
LO = np.r_[[0.3, 0.3, 0.0, -0.12, -0.12, -0.12, 0.60], np.full(36, -60.0), [-4.0, -0.4]]
HI = np.r_[[1.5, 1.5, 1.5, 0.12, 0.12, 0.12, 1.20], np.full(36, 60.0), [4.0, 0.4]]
EDGES = np.radians([-1e3, 60, 120, 180, 240, 300])


def launch_target(t: float, p: np.ndarray) -> np.ndarray:
    """Lopsided jump command at time t after the jump starts."""
    out = STANDV.copy()
    for g, (s, dt_) in {"L": (p[0], p[3]), "R": (p[1], p[4]), "C": (p[2], p[5])}.items():
        k = int(np.clip((t - dt_) / jump.CTRL_DT, 0, len(QCMD) - 1))
        idx = [i for i, j in enumerate(JOINTS) if GROUP[j] == g]
        out[idx] = STANDV[idx] + s * (QCMD[k, idx] - STANDV[idx])
    return np.clip(out, W.LLO, W.LHI)


def simulate(p, frame_cb=None, t_end=3.6):
    m, d = W._setup()
    p = np.asarray(p)
    mujoco.mj_resetData(m, d)
    d.qpos[W.QA] = W.HOMEV
    mujoco.mj_forward(m, d)
    sect = p[I_SECT:I_BAL].reshape(W.N_SECT, 6)
    sect_tgt = [W.target({L: W.STAR[L] + sect[k, i] for i, L in enumerate(W.LIMBS)})
                for k in range(W.N_SECT)]
    land = W.target(W.STANCE)
    t_switch = T_READY + p[6]
    dt = m.opt.timestep
    roll = prev = 0.0
    peak = bad_t = off_t = 0.0
    spread_sum = spread_n = sat = steps = 0
    airborne_peak = 0.0
    for step in range(int(t_end / dt)):
        t = step * dt
        phi = roll
        if t < T_READY:
            a = min(t / (T_READY - 0.2), 1.0)
            a = a * a * a * (10 - 15 * a + 6 * a * a)
            tgt = W.HOMEV + a * (STANDV - W.HOMEV)
        elif t < t_switch:
            tgt = launch_target(t - T_READY, p)
        elif phi < math.radians(330):
            tgt = sect_tgt[int(np.searchsorted(EDGES, phi, side="right") - 1)]
        else:
            tgt = land
        if t >= t_switch:                          # balance reflex once in the star
            xm = d.xmat[W.BASE]
            nose_up = math.asin(max(-1.0, min(1.0, xm[6])))
            c = p[I_BAL] * nose_up + p[I_BAL + 1] * (-d.qvel[4])
            tgt = tgt.copy()
            for L, i in W2._J1.items():
                down = c if L[1] == "R" else -c
                tgt[i] -= W2.SIDE[L] * down
            tgt = np.clip(tgt, W.LLO, W.LHI)
        tau = jump.torque(tgt, d.qpos[W.QA], d.qvel[W.VA])
        d.ctrl[W.AI] = tau
        mujoco.mj_step(m, d)
        if t >= T_READY:
            sat += int(np.sum(np.abs(tau) > 0.98 * 1.7464))
            steps += 1
            if d.ncon == 0 or all(W.FLOOR not in (c.geom1, c.geom2) for c in d.contact[:d.ncon]):
                airborne_peak = max(airborne_peak, d.xpos[W.BASE][2])
        x = d.xmat[W.BASE]
        ang = math.atan2(-x[5], x[8])
        if abs(x[6]) < 0.7:
            roll += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        peak = max(peak, roll)
        if 0.3 < roll < 6.0:
            for c in d.contact[:d.ncon]:
                g = c.geom1 if c.geom2 == W.FLOOR else c.geom2 if c.geom1 == W.FLOOR else -1
                if g >= 0 and W.BADG[g]:
                    bad_t += dt
                    break
            R = x.reshape(3, 3)
            bx = R[:, 0]
            dev = max(abs(math.atan2(bx[1], bx[0])), abs(math.asin(max(-1.0, min(1.0, bx[2])))))
            off_t += dt * max(0.0, dev - math.radians(15)) / W.OFF_LIMIT
            rel = (d.xpos[W.TIPB] - d.xpos[W.BASE]) @ R
            spread_sum += np.hypot(rel[:, 1], rel[:, 2]).mean()
            spread_n += 1
        if frame_cb:
            frame_cb(t, d)
    return dict(roll=roll, peak=peak, up=d.xmat[W.BASE][8], bad_t=bad_t, off_t=off_t,
                spread=spread_sum / max(spread_n, 1), sat=sat / max(steps * 22, 1),
                air_z=airborne_peak)


def score(p) -> float:
    r = simulate(p)
    turns = r["roll"] / (2 * math.pi)
    peak = r["peak"] / (2 * math.pi)
    s = 1.5 * min(peak, 1.0) + 1.0 * min(max(turns, 0), 1.0) - 1.5 * max(turns - 1.15, 0)
    s -= 6.0 * r["bad_t"] + 4.0 * r["off_t"]
    s += 1.0 * min(r["spread"] / 0.22, 1.0) * min(peak, 1.0)
    if turns > 0.85:
        s += 2.0 * max(r["up"], 0)
        if r["up"] > 0.9:
            s += 2.0
            if r["bad_t"] < 0.02 and r["off_t"] < 0.05:
                s += 2.0
    return s


def _cem(mean, sigma, free, iters, pop=48, elite=8, seed=0, floor=None):
    rng = np.random.default_rng(seed)
    base = mean.copy()
    best, best_s = mean.copy(), -1e9
    with Pool(2) as pool:
        for it in range(iters):
            cand = mean + sigma * rng.standard_normal((pop, NP))
            cand[:, ~free] = base[~free]
            cand = np.clip(cand, LO, HI)
            cand[0] = best
            scores = np.array(pool.map(score, list(cand)))
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > best_s:
                best_s, best = scores[order[0]], cand[order[0]].copy()
            el = cand[order[:elite]]
            mean = 0.7 * el.mean(0) + 0.3 * mean
            sigma = np.where(free, np.maximum(0.7 * el.std(0) + 0.3 * sigma, floor), 0.0)
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+5.0f}"
                  f"  peak {math.degrees(r['peak']):4.0f}  up {r['up']:+.2f}  air-height {r['air_z']:.3f}"
                  f"  shell {r['bad_t']:.2f}s  off-plane {r['off_t']:.2f}  spread {r['spread']:.3f}"
                  f"  motors-maxed {100 * r['sat']:.1f}%  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))
    return best


FLOOR = np.r_[[0.02] * 3, [0.004] * 3, [0.01], np.full(36, 2.0), [0.05, 0.005]]


def stage_launch():
    mean = np.zeros(NP)
    mean[:7] = [1.15, 0.7, 1.0, 0.0, 0.04, 0.0, 0.78]   # left side drives harder: spin toward +roll
    sigma = np.r_[[0.2, 0.2, 0.25, 0.04, 0.04, 0.04, 0.1], np.zeros(36), [1.5, 0.15]]
    free = np.zeros(NP, bool)
    free[:7] = True
    free[I_BAL:] = True
    best = _cem(mean, sigma, free, iters=25, seed=21, floor=FLOOR)
    LAUNCH.write_text(json.dumps(best.tolist()))


def stage_spokes():
    mean = np.array(json.loads(LAUNCH.read_text()))
    sigma = np.r_[[0.03] * 3, [0.006] * 3, [0.02], np.full(36, 25.0), [0.3, 0.03]]
    free = np.ones(NP, bool)
    _cem(mean, sigma, free, iters=40, seed=22, floor=FLOOR)


if __name__ == "__main__":
    {"launch": stage_launch, "spokes": stage_spokes}[sys.argv[1]]()
