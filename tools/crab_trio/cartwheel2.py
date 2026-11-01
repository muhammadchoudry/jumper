"""Stage 2 of the cartwheel search: from the flip, finish the roll.

Stage 1 (cartwheel.py, params in cw_stage1.json) gets the crab to throw itself
over sideways onto its back. Stage 2 keeps that motion and searches for what
the legs and claws do from there -- pushing against the floor while upside down
-- so the roll keeps going in the same direction to a full 360 and it lands
upright. Same motor model throughout (kp 20, kd 0.5, torque-speed envelope).

  python cartwheel2.py search
  python cartwheel2.py check
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

import cartwheel as c1
import jump
from stage import HOME, JOINTS

HERE = pathlib.Path(__file__).resolve().parent
STAGE1 = np.array(json.loads((HERE / "cw_stage1.json").read_text()))
PARAMS = HERE / "cartwheel2_params.json"

_S1 = c1.schedule(STAGE1)
T_HANDOFF = _S1[-1][0]          # when stage 1 releases to HOME (the flip happens after)
T_END = 4.6
DIRECTION = -1                  # stage 1 rolls toward negative roll

N_POSE = 3 * 2 * 7
LO = np.r_[np.full(N_POSE, -2.5), [0.0, 0.08, 0.08, 0.08]]
HI = np.r_[np.full(N_POSE, 2.5), [0.8, 0.7, 0.7, 0.7]]

_snap = None


def pose7(o: np.ndarray) -> np.ndarray:
    """o: (2 sides, 7) = leg J0, J1, J2, arm J0, J1, J2, J3 offsets from HOME."""
    tgt = dict(HOME)
    for i, (side, s) in enumerate((("L", 1.0), ("R", -1.0))):
        lj0, lj1, lj2, a0, a1, a2, a3 = o[i]
        for leg in c1.LEGS:
            tgt[f"{side}{leg}_J0_joint"] += s * lj0
            tgt[f"{side}{leg}_J1_joint"] += s * lj1
            tgt[f"{side}{leg}_J2_joint"] += s * lj2
        tgt[f"{side}F_J0_joint"] += s * a0
        tgt[f"{side}F_J1_joint"] += a1
        tgt[f"{side}F_J2_joint"] += s * a2
        tgt[f"{side}F_J3_joint"] += s * a3
    return np.array([tgt[j] for j in JOINTS])


def schedule(p: np.ndarray):
    poses = p[:N_POSE].reshape(3, 2, 7)
    s0, d1, d2, d3 = p[N_POSE:]
    t = T_HANDOFF + s0
    out = list(_S1[:-1]) + [(T_HANDOFF, None)]
    for k in range(3):
        out.append((t, pose7(poses[k])))
        t += (d1, d2, d3)[k]
    out.append((t, None))
    return out


def _snapshot():
    """Stage 1 is identical for every candidate up to the handoff: simulate it once."""
    global _snap
    if _snap is None:
        m, d = c1._setup()
        mujoco.mj_resetData(m, d)
        d.qpos[c1.QA] = c1.HOMEV
        mujoco.mj_forward(m, d)
        roll, prev = _run(m, d, _S1, 0.0, T_HANDOFF, 0.0, 0.0)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        st = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, st, spec)
        _snap = (st, roll, prev)
    return _snap


def _run(m, d, sched, t_start, t_end, roll, prev, frame_cb=None):
    global _peak
    _peak = DIRECTION * roll
    dt = m.opt.timestep
    for step in range(int(round(t_start / dt)), int(round(t_end / dt))):
        t = step * dt
        tgt = c1.HOMEV
        for t0, target in sched:
            if t >= t0 - 1e-9:
                tgt = c1.HOMEV if target is None else target
        tgt = np.clip(tgt, c1.LIM_LO, c1.LIM_HI)
        d.ctrl[c1.AI] = jump.torque(tgt, d.qpos[c1.QA], d.qvel[c1.VA])
        mujoco.mj_step(m, d)
        x = d.xmat[c1.BASE]
        ang = math.atan2(-x[5], x[8])
        roll += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        _peak = max(_peak, DIRECTION * roll)
        if frame_cb:
            frame_cb(t, d)
    return roll, prev


def simulate(p, frame_cb=None, t_end=T_END):
    m, d = c1._setup()
    if frame_cb is None:
        st, roll, prev = _snapshot()
        mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        t0 = T_HANDOFF
    else:  # full replay from the start, for filming
        mujoco.mj_resetData(m, d)
        d.qpos[c1.QA] = c1.HOMEV
        mujoco.mj_forward(m, d)
        roll = prev = t0 = 0.0
    roll, prev = _run(m, d, schedule(np.asarray(p)), t0, t_end, roll, prev, frame_cb)
    return dict(roll=roll, peak=_peak, up=d.xmat[c1.BASE][8], z=d.xpos[c1.BASE][2],
                y=d.xpos[c1.BASE][1], x=d.xpos[c1.BASE][0])


def score(p) -> float:
    r = simulate(p)
    turns = DIRECTION * r["roll"] / (2 * math.pi)
    peak = r["peak"] / (2 * math.pi)
    # credit for how far over it gets at its peak, even if it rocks back
    s = 1.5 * min(peak, 1.0) + 1.0 * min(max(turns, 0), 1.0) - 1.5 * max(turns - 1.15, 0)
    if turns > 0.85:
        s += 2.0 * max(r["up"], 0)
        if r["up"] > 0.9:
            s += 3.0
    return s


def search(iters=35, pop=40, elite=7, seed=1):
    rng = np.random.default_rng(seed)
    mean = np.r_[np.zeros(N_POSE), [0.2, 0.3, 0.3, 0.3]]
    sigma = np.r_[np.full(N_POSE, 0.9), [0.2, 0.15, 0.15, 0.15]]
    best, best_s = mean.copy(), -1e9
    with Pool(2) as pool:
        for it in range(iters):
            cand = np.clip(mean + sigma * rng.standard_normal((pop, len(LO))), LO, HI)
            cand[0] = best
            scores = np.array(pool.map(score, list(cand)))
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > best_s:
                best_s, best = scores[order[0]], cand[order[0]].copy()
            el = cand[order[:elite]]
            mean = 0.7 * el.mean(0) + 0.3 * mean
            sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma,
                               np.r_[np.full(N_POSE, 0.05), [0.01] * 4])
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+6.0f} deg"
                  f"  peak {math.degrees(r['peak']):4.0f}"
                  f"  up {r['up']:+.2f}  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))


if __name__ == "__main__":
    if sys.argv[1:] == ["search"]:
        search()
    else:
        print(simulate(np.array(json.loads(PARAMS.read_text()))))
