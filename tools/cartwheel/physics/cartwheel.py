"""Search for a sideways cartwheel that Jumper can do under physics.

The motion is an open-loop sequence of joint targets in four phases
(crouch -> push -> tuck -> catch -> recover). The targets are tracked with the
same motor model as the repo's jump (kp 20, kd 0.5, torque-speed envelope with a
1.75 N*m plateau), so the search can only find what the real motors could do.

Score: how far the body rolls about the forward axis (a full 360 is the goal)
plus a large bonus for finishing a full turn and ending upright.

  python cartwheel.py search            # cross-entropy search, writes cartwheel_params.json
  python cartwheel.py check             # replay the saved params and report
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

from tools.crab_trio import jump
from tools.crab_trio.stage import HOME, JOINTS, build

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "cartwheel_params.json"

LEGS = ("M", "R")      # walking legs: middle, rear
T_PUSH = 0.8           # push starts after this long crouching
T_END = 3.2

# 3 phases x 2 sides x 5 offsets, then 3 durations
N_POSE = 3 * 2 * 5
LO = np.r_[np.full(N_POSE, -2.0), [0.04, 0.04, 0.10]]
HI = np.r_[np.full(N_POSE, 2.0), [0.40, 0.70, 0.90]]

_model = _data = None


def _setup():
    global _model, _data, QA, VA, AI, BASE, LIM_LO, LIM_HI, HOMEV
    if _model is None:
        _model = build([(0.0, 0.0, 0.0)], torque_control=True)
        _data = mujoco.MjData(_model)
        QA = np.array([_model.jnt_qposadr[_model.joint("c0_" + j).id] for j in JOINTS])
        VA = np.array([_model.jnt_dofadr[_model.joint("c0_" + j).id] for j in JOINTS])
        AI = np.array([_model.actuator("c0_" + j).id for j in JOINTS])
        BASE = _model.body("c0_base_link").id
        LIM_LO = np.array([_model.jnt_range[_model.joint("c0_" + j).id][0] for j in JOINTS])
        LIM_HI = np.array([_model.jnt_range[_model.joint("c0_" + j).id][1] for j in JOINTS])
        HOMEV = np.array([HOME[j] for j in JOINTS])
    return _model, _data


def pose(offsets: np.ndarray) -> np.ndarray:
    """offsets: (2 sides, 5) = leg J1, leg J2, arm J1, arm J2, arm J3.

    Side-symmetric convention: positive leg offsets tuck/crouch, negative extend.
    The right side mirrors the left the same way the robot's HOME pose does.
    """
    tgt = dict(HOME)
    for side, s in (("L", 1.0), ("R", -1.0)):
        o = offsets[0 if side == "L" else 1]
        for leg in LEGS:
            tgt[f"{side}{leg}_J1_joint"] += s * o[0]
            tgt[f"{side}{leg}_J2_joint"] += s * o[1]
        tgt[f"{side}F_J1_joint"] += o[2]            # shoulder lift: same sign both sides
        tgt[f"{side}F_J2_joint"] += s * o[3]
        tgt[f"{side}F_J3_joint"] += s * o[4]
    return np.array([tgt[j] for j in JOINTS])


def schedule(p: np.ndarray):
    poses = p[:N_POSE].reshape(3, 2, 5)
    d1, d2, d3 = p[N_POSE:]
    crouch = pose(np.array([[0.35, 0.35, 0, 0, 0]] * 2))
    t1 = T_PUSH
    return [
        (0.0, crouch),
        (t1, pose(poses[0])),
        (t1 + d1, pose(poses[1])),
        (t1 + d1 + d2, pose(poses[2])),
        (t1 + d1 + d2 + d3, None),   # None = back to HOME
    ]


def simulate(p: np.ndarray, frame_cb=None, t_end=T_END):
    m, d = _setup()
    mujoco.mj_resetData(m, d)
    d.qpos[QA] = HOMEV
    mujoco.mj_forward(m, d)
    sched = schedule(np.asarray(p))
    dt = m.opt.timestep
    roll, prev = 0.0, 0.0
    peak_z = 0.0
    for step in range(int(t_end / dt)):
        t = step * dt
        tgt = HOMEV
        for t0, target in sched:
            if t >= t0:
                tgt = HOMEV if target is None else target
        tgt = np.clip(tgt, LIM_LO, LIM_HI)
        d.ctrl[AI] = jump.torque(tgt, d.qpos[QA], d.qvel[VA])
        mujoco.mj_step(m, d)
        x = d.xmat[BASE]
        ang = math.atan2(-x[5], x[8])          # roll of the body's up-axis about world x
        delta = (ang - prev + math.pi) % (2 * math.pi) - math.pi
        roll += delta
        prev = ang
        peak_z = max(peak_z, d.xpos[BASE][2])
        if frame_cb:
            frame_cb(t, d)
    up = d.xmat[BASE][8]
    return dict(roll=roll, up=up, z=d.xpos[BASE][2], peak_z=peak_z,
                y=d.xpos[BASE][1], x=d.xpos[BASE][0])


def score(p) -> float:
    r = simulate(p)
    turns = abs(r["roll"]) / (2 * math.pi)
    s = 2.0 * min(turns, 1.0) - 1.5 * max(turns - 1.15, 0)
    if turns > 0.85:
        s += 2.0 * max(r["up"], 0)            # land it
        if r["up"] > 0.9:
            s += 3.0
    return s


def search(iters=40, pop=32, elite=6, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.zeros(len(LO))
    # a hand guess to start from: left legs shove, right side tucks, arms swing over
    g = np.zeros((3, 2, 5))
    g[0, 0] = [-0.9, -0.9, -0.8, 0.0, 0.0]     # push: left extends
    g[0, 1] = [0.8, 0.8, -1.5, 0.0, 0.0]       # right tucks, right claw reaches over
    g[1, :] = [0.9, 0.9, -1.5, 0.0, 0.0]       # tuck everything mid-roll
    g[2, :] = [-0.3, -0.3, 0.0, 0.0, 0.0]      # reach out to land
    mean[:N_POSE] = g.ravel()
    mean[N_POSE:] = [0.15, 0.3, 0.4]
    sigma = np.r_[np.full(N_POSE, 0.7), [0.08, 0.15, 0.2]]
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
            sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, np.r_[np.full(N_POSE, 0.05), [0.01] * 3])
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+6.0f} deg"
                  f"  up {r['up']:+.2f}  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))
    return best


if __name__ == "__main__":
    if sys.argv[1:] == ["search"]:
        search()
    else:
        p = np.array(json.loads(PARAMS.read_text()))
        print(simulate(p))
