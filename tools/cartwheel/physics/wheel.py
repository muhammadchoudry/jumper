"""A true cartwheel for Jumper: limbs spread like wheel spokes, rolling over the
leg tips (left legs -> claws over the top -> right legs -> feet) with the shell
never touching the floor.

Controller: after a push-off, the joint targets depend on the body's measured
roll angle (an IMU reading on the real robot): the roll is cut into six 60-degree
sectors, each with its own small extension of every limb on top of a spread
'star' pose, so a spoke can shove as it comes round to the floor. Same motor
model as the robot's jump (kp 20, kd 0.5, torque-speed envelope, 1.75 N*m).

  python wheel.py search
  python wheel.py check
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
PARAMS = HERE / "wheel_params.json"
STAR_FILE = HERE / "wheel_star.json"

LIMBS = ("LF", "RF", "LM", "RM", "LR", "RR")
TIPS = {"LF": "LF_finger_tip_link", "RF": "RF_finger_tip_link", "LM": "LM_foot_tip_link",
        "RM": "RM_foot_tip_link", "LR": "LR_foot_tip_link", "RR": "RR_foot_tip_link"}
# spoke directions in the roll plane, degrees from the left (+y) toward up (+z)
# A 'thick' wheel like a human cartwheel: feet in pairs spread front-to-back for
# pitch support (left pair, right pair), claws as the hands over the top.
# Six spokes ~60 deg apart round the roll plane, alternating in front of and
# behind the body so that whenever two spokes share the floor they brace pitch.
SPOKES = {"LM": -45, "LR": 12, "LF": 68, "RF": 115, "RR": 170, "RM": 228}
X_TARGET = {"LM": 0.10, "LR": -0.12, "LF": 0.10, "RF": -0.06, "RR": 0.05, "RM": -0.12}
# two joints per limb that the search may bend: (lift, extend)
KNOBS = {L: ((f"{L}_J1_joint", f"{L}_J3_joint") if L[1] == "F" else (f"{L}_J1_joint", f"{L}_J2_joint"))
         for L in LIMBS}
SIDE = {L: (1.0 if L[0] == "L" else -1.0) for L in LIMBS}
NONLIMB = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link", "camera_link")

T_CROUCH = 0.8
T_END = 3.8
DIRECTION = 1                     # tip over onto the left legs first
N_SECT = 6
OFF_LIMIT = math.radians(30)      # a cartwheel stays in its plane: no twisting, no nose-dive

# params: push (6 limbs x 2) + push duration, star tweak (6 x 2), sectors (6 x 6 extensions)
I_PUSH, I_DPUSH, I_STAR, I_SECT = 0, 12, 13, 25
NP = 25 + N_SECT * 6
LO = np.r_[np.full(12, -1.8), [0.06], np.full(12, -0.6), np.full(N_SECT * 6, -1.2)]
HI = np.r_[np.full(12, 1.8), [0.45], np.full(12, 0.6), np.full(N_SECT * 6, 1.2)]

_m = _d = None


def _setup():
    global _m, _d, QA, VA, AI, BASE, LLO, LHI, HOMEV, TIPB, FLOOR, BADG, JIDX
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
        JIDX = {j: i for i, j in enumerate(JOINTS)}
    return _m, _d


def find_star() -> dict[str, float]:
    """For each limb, the joint angles that push its tip furthest out along its spoke."""
    m, d = _setup()
    rng = np.random.default_rng(0)
    star = dict(HOME)
    for L in LIMBS:
        js = [j for j in JOINTS if j.startswith(L) and not j.endswith("J4_joint")]
        want = math.radians(SPOKES[L])
        best, best_r = None, -1
        for _ in range(20000):
            d.qpos[:] = 0
            d.qpos[3] = 1
            d.qpos[QA] = HOMEV
            vals = {j: rng.uniform(*m.jnt_range[m.joint("c0_" + j).id]) for j in js}
            for j, v in vals.items():
                d.qpos[QA[JIDX[j]]] = v
            mujoco.mj_kinematics(m, d)
            p = d.xpos[m.body("c0_" + TIPS[L]).id] - d.xpos[BASE]
            ang = math.atan2(p[2], p[1])
            err = abs((ang - want + math.pi) % (2 * math.pi) - math.pi)
            r = math.hypot(p[1], p[2]) - 3.0 * abs(p[0] - X_TARGET[L])
            if err < math.radians(6) and r > best_r:
                best, best_r = vals, r
        star.update(best)
        print(f"{L}: spoke {SPOKES[L]:4d} deg, score {best_r:.3f}")
    STAR_FILE.write_text(json.dumps(star))
    return star


STAR = json.loads(STAR_FILE.read_text()) if STAR_FILE.exists() else None


def _apply(base: dict, offs: np.ndarray) -> np.ndarray:
    """offs: (6 limbs, 2) offsets on each limb's (lift, extend) joints, mirrored per side."""
    v = np.array([base[j] for j in JOINTS])
    for i, L in enumerate(LIMBS):
        for k, j in enumerate(KNOBS[L]):
            sgn = 1.0 if (L[1] == "F" and k == 0) else SIDE[L]   # claw shoulder lift is unmirrored
            v[JIDX[j]] += sgn * offs[i, k]
    return v


def simulate(p, frame_cb=None, t_end=T_END):
    m, d = _setup()
    p = np.asarray(p)
    mujoco.mj_resetData(m, d)
    d.qpos[QA] = HOMEV
    mujoco.mj_forward(m, d)
    star = _apply(STAR, p[I_STAR:I_STAR + 12].reshape(6, 2))
    # crouch with the legs under the body (where they are strongest), push off
    # sideways to build momentum, then open out into the spread star to roll
    crouch = _apply(HOME, np.array([[0, 0]] * 2 + [[0.35, 0.35]] * 4, dtype=float))
    push = _apply(HOME, p[I_PUSH:I_PUSH + 12].reshape(6, 2))
    t_push_end = T_CROUCH + p[I_DPUSH]
    sect = p[I_SECT:].reshape(N_SECT, 6)
    sect_tgt = [star + _apply({j: 0.0 for j in JOINTS},
                              np.stack([np.zeros(6), sect[k]], 1)) for k in range(N_SECT)]
    dt = m.opt.timestep
    roll = prev = 0.0
    peak = 0.0
    bad_t = 0.0
    off_t = 0.0                                       # time spent twisting or pitching
    spread_sum = spread_n = 0
    for step in range(int(t_end / dt)):
        t = step * dt
        phi = DIRECTION * roll
        if t < T_CROUCH:             # ease from the normal stance into the splay
            a = min(t / (T_CROUCH - 0.2), 1.0)
            a = a * a * a * (10 - 15 * a + 6 * a * a)
            tgt = HOMEV + a * (crouch - HOMEV)
        elif t < t_push_end:
            tgt = push
        elif phi < 335 * math.pi / 180:
            k = int(np.clip(phi // (math.pi / 3), 0, N_SECT - 1))
            tgt = sect_tgt[k]
        else:
            tgt = HOMEV                               # feet down for the landing
        tgt = np.clip(tgt, LLO, LHI)
        d.ctrl[AI] = jump.torque(tgt, d.qpos[QA], d.qvel[VA])
        mujoco.mj_step(m, d)
        x = d.xmat[BASE]
        ang = math.atan2(-x[5], x[8])
        if abs(x[6]) < 0.7:          # |pitch| < ~45 deg: roll is well defined
            roll += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        phi = DIRECTION * roll
        peak = max(peak, phi)
        if 0.3 < phi < 6.0:          # mid-cartwheel: score shell contact and spread
            for c in d.contact[:d.ncon]:
                g = c.geom1 if c.geom2 == FLOOR else c.geom2 if c.geom1 == FLOOR else -1
                if g >= 0 and BADG[g]:
                    bad_t += dt
                    break
            R = x.reshape(3, 3)
            bx = R[:, 0]                                  # body forward axis in the world
            heading = abs(math.atan2(bx[1], bx[0]))
            pitch = abs(math.asin(max(-1.0, min(1.0, bx[2]))))
            off_t += dt * max(0.0, max(heading, pitch) - math.radians(15)) / OFF_LIMIT
            rel = (d.xpos[TIPB] - d.xpos[BASE]) @ R    # tips in the body frame
            spread_sum += np.hypot(rel[:, 1], rel[:, 2]).mean()
            spread_n += 1
        if frame_cb:
            frame_cb(t, d)
    return dict(roll=roll, peak=peak, up=d.xmat[BASE][8], bad_t=bad_t, off_t=off_t,
                spread=spread_sum / max(spread_n, 1), y=d.xpos[BASE][1])


def score(p) -> float:
    r = simulate(p)
    turns = DIRECTION * r["roll"] / (2 * math.pi)
    peak = r["peak"] / (2 * math.pi)
    s = 1.5 * min(peak, 1.0) + 1.0 * min(max(turns, 0), 1.0) - 1.5 * max(turns - 1.15, 0)
    s -= 6.0 * r["bad_t"]                              # shell on the floor
    s -= 4.0 * r["off_t"]                              # twisting / pitching out of the wheel plane
    s += 1.0 * min(r["spread"] / 0.24, 1.0) * min(peak, 1.0)
    if turns > 0.85:
        s += 2.0 * max(r["up"], 0)
        if r["up"] > 0.9:
            s += 2.0
            if r["bad_t"] < 0.02 and r["off_t"] < 0.05:
                s += 2.0                               # a clean wheel: shell never down
    return s


def search(iters=60, pop=48, elite=8, seed=5):
    rng = np.random.default_rng(seed)
    mean = np.zeros(NP)
    # push guess: left side crouches, right side shoves -> body tips onto the left
    push = np.zeros((6, 2))
    for i, L in enumerate(LIMBS):
        if L[1] != "F":
            push[i] = [0.6, 0.6] if L[0] == "L" else [-0.9, -0.9]
    mean[I_PUSH:I_PUSH + 12] = push.ravel()
    mean[I_DPUSH] = 0.18
    sigma = np.r_[np.full(12, 0.6), [0.08], np.full(12, 0.25), np.full(N_SECT * 6, 0.5)]
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
            sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.03)
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+5.0f}"
                  f"  peak {math.degrees(r['peak']):4.0f}  up {r['up']:+.2f}"
                  f"  shell-down {r['bad_t']:.2f}s  off-plane {r['off_t']:.2f}s  spread {r['spread']:.3f}"
                  f"  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))


if __name__ == "__main__":
    if sys.argv[1:] == ["star"]:
        find_star()
    elif sys.argv[1:] == ["search"]:
        search()
    else:
        r = simulate(np.array(json.loads(PARAMS.read_text())))
        print({k: round(float(v), 3) for k, v in r.items()})
