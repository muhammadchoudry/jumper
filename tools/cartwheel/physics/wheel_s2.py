"""Straight-spoke cartwheel, launched from a crouch.

The wide straight stance holds easily but can't throw the body over (rolling
over its wide base needs ~3 J). So: crouch with the legs folded under the body
-- where the motors have leverage, as in the repo's jump -- shove off sideways,
then snap the limbs out into straight spokes and roll on them. The spoke
schedule, motor model and scoring are the same as wheel_s.py.

  python wheel_s2.py search
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
from tools.cartwheel.physics import wheel_s as W
from tools.crab_trio.stage import HOME, JOINTS

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "wheel_s2_params.json"

T_CROUCH = 1.0
CROUCH = (0.8, -1.6)    # hip lift, knee fold (left-side sense) from the straight stance
LIMBS = W.LIMBS
SIDE = {L: (1.0 if L[0] == "L" else -1.0) for L in LIMBS}
KNOBS = {L: ((f"{L}_J1_joint", f"{L}_J3_joint") if L[1] == "F" else (f"{L}_J1_joint", f"{L}_J2_joint"))
         for L in LIMBS}

# params: push (6 limbs x (lift, extend)) + push duration + sector spoke offsets (6 x 6)
#         + balance reflex gains (pitch, pitch rate)
I_PUSH, I_DUR, I_SECT = 0, 12, 13
I_BAL = 13 + 6 * W.N_SECT
NP = I_BAL + 2
LO = np.r_[np.full(12, -1.8), [0.06], np.full(6 * W.N_SECT, -60.0), [-4.0, -0.4]]
HI = np.r_[np.full(12, 1.8), [0.45], np.full(6 * W.N_SECT, 60.0), [4.0, 0.4]]

# sector edges (rad): the 4th sector starts at 165 deg so it is active while the crab
# rests in its upside-down 'handstand' (~180 deg) and can push on from there
EDGES = np.radians([-1e3, 60, 120, 165, 240, 300])

# balance reflex: middle legs and rear legs push in opposition to front-back tipping
_J1 = {L: JOINTS.index(f"{L}_J1_joint") for L in ("LM", "RM", "LR", "RR")}


def _offset(base: np.ndarray, offs: np.ndarray) -> np.ndarray:
    v = base.copy()
    for i, L in enumerate(LIMBS):
        for k, j in enumerate(KNOBS[L]):
            sgn = 1.0 if (L[1] == "F" and k == 0) else SIDE[L]
            v[JOINTS.index(j)] += sgn * offs[i, k]
    return np.clip(v, W.LLO, W.LHI)


def simulate(p, frame_cb=None, t_end=W.T_END, direction=1):
    m, d = W._setup()
    p = np.asarray(p)
    mujoco.mj_resetData(m, d)
    d.qpos[W.QA] = W.HOMEV
    mujoco.mj_forward(m, d)
    # crouch in the fanned stance (balanced front-to-back), knees folded under the body
    stance = W.target(W.STANCE)
    crouch = _offset(stance, np.array([[0, 0]] * 2 + [list(CROUCH)] * 4, dtype=float))
    push = _offset(crouch, p[I_PUSH:I_PUSH + 12].reshape(6, 2))
    t_push = T_CROUCH + p[I_DUR]
    sect = p[I_SECT:I_BAL].reshape(W.N_SECT, 6)
    sect_tgt = [W.target({L: W.STAR[L] + sect[k, i] for i, L in enumerate(LIMBS)})
                for k in range(W.N_SECT)]
    land = W.target(W.STANCE)
    dt = m.opt.timestep
    roll = prev = 0.0
    peak = bad_t = off_t = 0.0
    spread_sum = spread_n = sat = steps = 0
    for step in range(int(t_end / dt)):
        t = step * dt
        phi = direction * roll
        if t < T_CROUCH:
            a = min(t / (T_CROUCH - 0.2), 1.0)
            a = a * a * a * (10 - 15 * a + 6 * a * a)
            tgt = W.HOMEV + a * (crouch - W.HOMEV)
        elif t < t_push:
            tgt = push
        elif phi < math.radians(330):
            tgt = sect_tgt[int(np.searchsorted(EDGES, phi, side="right") - 1)]
        else:
            tgt = land
        if t >= T_CROUCH:
            xm = d.xmat[W.BASE]
            nose_up = math.asin(max(-1.0, min(1.0, xm[6])))
            c = p[I_BAL] * nose_up + p[I_BAL + 1] * (-d.qvel[4])
            tgt = tgt.copy()
            for L, i in _J1.items():                 # spoke 'down' is -J1 on the left, +J1 on the right
                down = c if L[1] == "R" else -c
                tgt[i] -= SIDE[L] * down
            tgt = np.clip(tgt, W.LLO, W.LHI)
        tau = jump.torque(tgt, d.qpos[W.QA], d.qvel[W.VA])
        d.ctrl[W.AI] = tau
        mujoco.mj_step(m, d)
        if t >= T_CROUCH:
            sat += int(np.sum(np.abs(tau) > 0.98 * 1.7464))
            steps += 1
        x = d.xmat[W.BASE]
        ang = math.atan2(-x[5], x[8])
        if abs(x[6]) < 0.7:
            roll += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        phi = direction * roll
        peak = max(peak, phi)
        if 0.3 < phi < 6.0:
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
    return dict(roll=roll * direction, peak=peak, up=d.xmat[W.BASE][8], bad_t=bad_t, off_t=off_t,
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


def search(iters=60, pop=48, elite=8, seed=11, launch=None):
    rng = np.random.default_rng(seed)
    mean = np.zeros(NP)
    if launch is not None:          # stage 2: keep a found launch, search the rolling spokes
        mean[:13] = launch[:13]
        mean[I_BAL:] = launch[13:15]
    # push guess (rolls toward +roll = left side up): left legs shove, right side folds
    push = np.zeros((6, 2))
    for i, L in enumerate(LIMBS):
        if L[1] != "F":
            push[i] = [-1.2, 1.6] if L[0] == "L" else [0.2, -0.3]   # left legs straighten hard
    if launch is None:
        mean[I_PUSH:I_PUSH + 12] = push.ravel()
        mean[I_DUR] = 0.15
        sigma = np.r_[np.full(12, 0.8), [0.1], np.full(6 * W.N_SECT, 25.0), [1.5, 0.15]]
    else:
        sigma = np.r_[np.full(12, 0.08), [0.01], np.full(6 * W.N_SECT, 25.0), [0.2, 0.02]]
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
            sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma,
                               np.r_[np.full(12, 0.05), [0.01], np.full(6 * W.N_SECT, 2.0), [0.05, 0.005]])
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+5.0f}"
                  f"  peak {math.degrees(r['peak']):4.0f}  up {r['up']:+.2f}  shell {r['bad_t']:.2f}s"
                  f"  off-plane {r['off_t']:.2f}  spread {r['spread']:.3f}  motors-maxed {100 * r['sat']:.1f}%"
                  f"  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))


def search3(iters=50, pop=48, elite=8, seed=12):
    """Stage 3: keep launch + first-half spokes (gets to a clean handstand); search the
    spokes for the second half (sectors 3-5) and the landing reflex gains."""
    rng = np.random.default_rng(seed)
    base = np.array(json.loads((HERE / "ws2_stage2.json").read_text()))
    free = np.zeros(NP, bool)
    free[I_SECT + 3 * 6:I_BAL] = True                 # sectors 3, 4, 5
    free[I_BAL:] = True
    mean = base.copy()
    sigma = np.where(free, 0.0, 0.0)
    sigma[I_SECT + 3 * 6:I_BAL] = 35.0
    sigma[I_BAL:] = [0.6, 0.06]
    best, best_s = base.copy(), -1e9
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
            sigma = np.where(free, np.maximum(0.7 * el.std(0) + 0.3 * sigma, 1.0), 0.0)
            sigma[I_BAL:] = np.maximum(sigma[I_BAL:], [0.05, 0.005])
            r = simulate(best)
            print(f"iter {it:2d}  best {best_s:5.2f}  roll {math.degrees(r['roll']):+5.0f}"
                  f"  peak {math.degrees(r['peak']):4.0f}  up {r['up']:+.2f}  shell {r['bad_t']:.2f}s"
                  f"  off-plane {r['off_t']:.2f}  spread {r['spread']:.3f}  motors-maxed {100 * r['sat']:.1f}%"
                  f"  gen-mean {scores.mean():5.2f}", flush=True)
            PARAMS.write_text(json.dumps(best.tolist()))


if __name__ == "__main__":
    if sys.argv[1:] == ["search"]:
        search()
    elif sys.argv[1:] == ["search2"]:
        search(launch=np.load(HERE / "launch_best.npy"))
    elif sys.argv[1:] == ["search3"]:
        search3()
    else:
        r = simulate(np.array(json.loads(PARAMS.read_text())))
        print({k: round(float(v), 3) for k, v in r.items()})
