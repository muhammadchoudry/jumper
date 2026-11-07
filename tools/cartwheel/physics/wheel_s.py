"""Straight-spoke cartwheel: limbs locked straight, load carried along the bones.

From the wide straight-leg stance (legs fanned, claws angled forward -- held at
~55% of motor torque), the crab shoves off and rolls sideways. During the roll
the controller only re-aims each straight limb (hip angle in the roll plane);
knees stay straight, so the ground force runs along the limbs instead of
fighting the motors. Targets are scheduled on the measured roll angle (an IMU
reading on the real robot) in six 60-degree sectors.

Motor model throughout: kp 20, kd 0.5, torque-speed envelope, 1.75 N*m plateau
(the repo's jump model). The score also reports how often motors saturate.

  python wheel_s.py search
  python wheel_s.py check
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
from tools.cartwheel.physics import straight as S
from tools.crab_trio.stage import HOME, JOINTS, build

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "wheel_s_params.json"

LIMBS = ("LF", "RF", "LM", "RM", "LR", "RR")
YAW = {"LF": 25, "RF": 25, "LM": 30, "RM": 30, "LR": -30, "RR": -30}
STANCE = {"LM": -50, "LR": -50, "RM": 230, "RR": 230, "LF": 60, "RF": 120}
# a spread 'star' for rolling: spokes spaced round the circle (deg, from the left, up = 90)
STAR = {"RM": 225, "RR": 165, "RF": 120, "LF": 60, "LR": 15, "LM": -45}
TIPS = {"LF": "LF_finger_tip_link", "RF": "RF_finger_tip_link", "LM": "LM_foot_tip_link",
        "RM": "RM_foot_tip_link", "LR": "LR_foot_tip_link", "RR": "RR_foot_tip_link"}
NONLIMB = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link", "camera_link")

T_STANCE = 1.4          # ease from the normal stance into the wide straight stance
T_END = 4.2
N_SECT = 6
OFF_LIMIT = math.radians(30)

# params: push spoke offsets (6) + push duration + sector spoke offsets (6 x 6)
I_PUSH, I_DUR, I_SECT = 0, 6, 7
NP = 7 + 6 * N_SECT
LO = np.r_[np.full(6, -70.0), [0.06], np.full(6 * N_SECT, -60.0)]
HI = np.r_[np.full(6, 70.0), [0.5], np.full(6 * N_SECT, 60.0)]

_m = _d = None


def _setup():
    global _m, _d, QA, VA, AI, BASE, LLO, LHI, HOMEV, TIPB, FLOOR, BADG
    if _m is None:
        _m = build([(0.0, 0.0, 0.0)], torque_control=True)
        _d = mujoco.MjData(_m)
        QA = np.array([_m.jnt_qposadr[_m.joint("c0_" + j).id] for j in JOINTS])
        VA = np.array([_m.jnt_dofadr[_m.joint("c0_" + j).id] for j in JOINTS])
        AI = np.array([_m.actuator("c0_" + j).id for j in JOINTS])
        BASE = _m.body("c0_base_link").id
        LLO = np.array([_m.jnt_range[_m.joint("c0_" + j).id][0] for j in JOINTS])
        LHI = np.array([_m.jnt_range[_m.joint("c0_" + j).id][1] for j in JOINTS])
        HOMEV = np.array([HOME[j] for j in JOINTS])
        TIPB = np.array([_m.body("c0_" + TIPS[L]).id for L in LIMBS])
        FLOOR = _m.geom("floor").id
        bad = {_m.body("c0_" + b).id for b in NONLIMB}
        BADG = np.array([_m.geom_bodyid[g] in bad for g in range(_m.ngeom)])
    return _m, _d


def target(spokes: dict[str, float]) -> np.ndarray:
    p = S.pose({L: (spokes[L], YAW[L]) for L in LIMBS})
    return np.clip(np.array([p[j] for j in JOINTS]), LLO, LHI)


def simulate(p, frame_cb=None, t_end=T_END, direction=1):
    m, d = _setup()
    p = np.asarray(p)
    mujoco.mj_resetData(m, d)
    d.qpos[QA] = HOMEV
    mujoco.mj_forward(m, d)
    stance = target(STANCE)
    push = target({L: STANCE[L] + p[I_PUSH + i] for i, L in enumerate(LIMBS)})
    t_push = T_STANCE + p[I_DUR]
    sect = p[I_SECT:].reshape(N_SECT, 6)
    sect_tgt = [target({L: STAR[L] + sect[k, i] for i, L in enumerate(LIMBS)}) for k in range(N_SECT)]
    land = stance
    dt = m.opt.timestep
    roll = prev = 0.0
    peak = bad_t = off_t = 0.0
    spread_sum = spread_n = 0
    sat = steps = 0
    for step in range(int(t_end / dt)):
        t = step * dt
        phi = direction * roll
        if t < T_STANCE:
            a = min(t / (T_STANCE - 0.3), 1.0)
            a = a * a * a * (10 - 15 * a + 6 * a * a)
            tgt = HOMEV + a * (stance - HOMEV)
        elif t < t_push:
            tgt = push
        elif phi < math.radians(330):
            tgt = sect_tgt[int(np.clip(phi // (math.pi / 3), 0, N_SECT - 1))]
        else:
            tgt = land
        tau = jump.torque(tgt, d.qpos[QA], d.qvel[VA])
        d.ctrl[AI] = tau
        mujoco.mj_step(m, d)
        if t >= T_STANCE:
            sat += int(np.sum(np.abs(tau) > 0.98 * 1.7464))
            steps += 1
        x = d.xmat[BASE]
        ang = math.atan2(-x[5], x[8])
        if abs(x[6]) < 0.7:
            roll += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        phi = direction * roll
        peak = max(peak, phi)
        if 0.3 < phi < 6.0:
            for c in d.contact[:d.ncon]:
                g = c.geom1 if c.geom2 == FLOOR else c.geom2 if c.geom1 == FLOOR else -1
                if g >= 0 and BADG[g]:
                    bad_t += dt
                    break
            R = x.reshape(3, 3)
            bx = R[:, 0]
            dev = max(abs(math.atan2(bx[1], bx[0])), abs(math.asin(max(-1.0, min(1.0, bx[2])))))
            off_t += dt * max(0.0, dev - math.radians(15)) / OFF_LIMIT
            rel = (d.xpos[TIPB] - d.xpos[BASE]) @ R
            spread_sum += np.hypot(rel[:, 1], rel[:, 2]).mean()
            spread_n += 1
        if frame_cb:
            frame_cb(t, d)
    return dict(roll=roll * direction, peak=peak, up=d.xmat[BASE][8], bad_t=bad_t, off_t=off_t,
                spread=spread_sum / max(spread_n, 1), sat=sat / max(steps * 22, 1))


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


def search(iters=60, pop=48, elite=8, seed=7):
    rng = np.random.default_rng(seed)
    mean = np.zeros(NP)
    # push guess: the left legs drive down (lifting the left side), right legs brace
    mean[I_PUSH:I_PUSH + 6] = [0, 0, -40, 0, -40, 0]
    mean[I_DUR] = 0.2
    sigma = np.r_[np.full(6, 30.0), [0.08], np.full(6 * N_SECT, 25.0)]
    best, best_s = mean.copy(), -1e9
    with Pool(2) as pool:
        for it in range(iters):
            cand = np.clip(mean + sigma * rng.standard_normal((pop, NP)), LO, HI)
            cand[0] = best
            scores = np.array(pool.map(score, list(cand)))
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > best_s:
                best_s, best = scores[order[0]], cand[order[0]].copy()
            el = cand[order[:elite]]
            mean = 0.7 * el.mean(0) + 0.3 * mean
            sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, np.r_[np.full(6, 2.0), [0.01], np.full(6 * N_SECT, 2.0)])
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+5.0f}"
                  f"  peak {math.degrees(r['peak']):4.0f}  up {r['up']:+.2f}  shell {r['bad_t']:.2f}s"
                  f"  off-plane {r['off_t']:.2f}  spread {r['spread']:.3f}  motors-maxed {100 * r['sat']:.1f}%"
                  f"  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))


if __name__ == "__main__":
    if sys.argv[1:] == ["search"]:
        search()
    else:
        r = simulate(np.array(json.loads(PARAMS.read_text())))
        print({k: round(float(v), 3) for k, v in r.items()})
