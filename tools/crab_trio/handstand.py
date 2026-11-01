"""Claw handstand and vertical push-ups for Jumper, under physics.

Geometry: with the claw hip-yaw J0 at -90 deg (left) / +90 deg (right), each claw
points straight forward along the body, and its J2 (elbow), J3 (wrist) and J4
(jaw) all bend in the body's pitch plane. Pitched nose-down to vertical, the
straight claws then carry the body's weight along their length (little motor
torque), and J2/J3 bending gives the push-up. Opening the jaw while tilting the
wrist half as much turns each claw end into a two-point foot (palm pad + finger
tip) spread front-to-back, so the four contact points form a real support area.

Motor model throughout: jump.torque (kp 20, kd 0.5, torque-speed envelope,
1.75 N*m plateau) on every joint.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

import jump
from stage import HOME, JOINTS, build

PEAK = 1.7464
_m = _d = None


def setup():
    global _m, _d, QA, VA, AI, BASE, LLO, LHI, HOMEV, FLOOR, CLAWG, JI
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
        FLOOR = _m.geom("floor").id
        claw_bodies = {i for i in range(_m.nbody)
                       if _m.body(i).name.startswith(("c0_LF_", "c0_RF_"))}
        CLAWG = np.array([_m.geom_bodyid[g] in claw_bodies for g in range(_m.ngeom)])
        JI = {j: i for i, j in enumerate(JOINTS)}
    return _m, _d


def claws(elbow=0.0, wrist=0.0, jaw=0.0, yaw=90.0, spread=0.0, base=None) -> dict:
    """Claws pointing forward. elbow/wrist/jaw in rad, + = bend toward the belly
    (left-claw sense; the right claw mirrors). jaw opens the pincer; the wrist is
    then tilted by -jaw/2 on top so palm and finger tip share the floor."""
    p = dict(base or HOME)
    y = math.radians(yaw)
    for s, sg in (("L", 1.0), ("R", -1.0)):
        p[f"{s}F_J0_joint"] = -sg * y
        p[f"{s}F_J1_joint"] = spread
        p[f"{s}F_J2_joint"] = sg * elbow
        p[f"{s}F_J3_joint"] = sg * (wrist - jaw / 2)
        p[f"{s}F_J4_joint"] = -sg * jaw
    return p


def vec(p: dict) -> np.ndarray:
    return np.clip(np.array([p[j] for j in JOINTS]), LLO, LHI)


def body_axes(d):
    R = d.xmat[BASE].reshape(3, 3)
    return R[:, 0], R[:, 1], R[:, 2]


def nose_down_deg(d) -> float:
    """Pitch of the body's nose below horizontal (90 = vertical handstand)."""
    bx, _, _ = body_axes(d)
    return math.degrees(math.asin(max(-1.0, min(1.0, -bx[2]))))


def floor_contacts(d):
    """(claw contact count, any non-claw body part on the floor)."""
    m = _m
    n_claw, other = 0, False
    for c in d.contact[:d.ncon]:
        g = c.geom1 if c.geom2 == FLOOR else c.geom2 if c.geom1 == FLOOR else -1
        if g < 0:
            continue
        if CLAWG[g]:
            n_claw += 1
        else:
            other = True
    return n_claw, other


def place_handstand(target: np.ndarray, pitch_deg=90.0, clearance=0.003):
    """Put the robot kinematically into a nose-down pose just touching the floor."""
    m, d = setup()
    mujoco.mj_resetData(m, d)
    a = math.radians(pitch_deg) / 2
    d.qpos[3:7] = [math.cos(a), 0, math.sin(a), 0]
    d.qpos[QA] = target
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    low = min(d.geom_xpos[g][2] - (m.geom_rbound[g] if False else 0) for g in range(m.ngeom) if CLAWG[g])
    # lowest claw collision vertex
    zmin = 9.0
    for g in range(m.ngeom):
        if not CLAWG[g] or not m.geom_contype[g] or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        zmin = min(zmin, float(((d.geom_xmat[g].reshape(3, 3) @ v.T)[2] + d.geom_xpos[g][2]).min()))
    d.qpos[2] = 1.0 - zmin + clearance
    mujoco.mj_forward(m, d)
    del low
    return m, d


def step(target: np.ndarray, record=None):
    m, d = _m, _d
    tau = jump.torque(np.clip(target, LLO, LHI), d.qpos[QA], d.qvel[VA])
    d.ctrl[AI] = tau
    mujoco.mj_step(m, d)
    if record is not None:
        record.append(np.abs(tau))
    return tau


# ── Handstand pose, push-ups and the kick-up entry ──────────────────────────
HS = dict(elbow=0.5, wrist=0.1, jaw=0.6)        # statically balanced vertical handstand
REFLEX_KP = 4.5                                  # wrist (J3) 'ankle' reflex on nose-down angle
SETPOINT = 88.0
LEGS4 = ("LM", "RM", "LR", "RR")


def legs_offset(base: np.ndarray, m_lift, m_ext, r_lift, r_ext) -> np.ndarray:
    """Side-symmetric J1/J2 offsets for middle and rear legs (+ = tuck, - = extend)."""
    v = base.copy()
    for L in LEGS4:
        s = 1.0 if L[0] == "L" else -1.0
        lift, ext = (m_lift, m_ext) if L[1] == "M" else (r_lift, r_ext)
        v[JI[f"{L}_J1_joint"]] += s * lift
        v[JI[f"{L}_J2_joint"]] += s * ext
    return v


def reflex(tgt: np.ndarray, d, kp=REFLEX_KP, setpoint=SETPOINT, kd=0.0) -> np.ndarray:
    tgt = tgt.copy()
    c = kp * math.radians(nose_down_deg(d) - setpoint) + kd * d.qvel[4]
    tgt[JI["LF_J3_joint"]] += c
    tgt[JI["RF_J3_joint"]] -= c
    return tgt


# entry params: plant claws (elbow, wrist), kick legs (m_lift, m_ext, r_lift, r_ext),
# kick time, tuck legs (m_lift, m_ext, r_lift, r_ext), plant time, reflex-on angle
ENTRY_LO = np.array([-1.2, -1.2, -1.5, -1.5, -1.5, -1.5, 0.06, -1.0, -1.5, -1.0, -1.5, 0.3, 40.0])
ENTRY_HI = np.array([1.2, 1.2, 1.5, 1.5, 1.5, 1.5, 0.45, 1.5, 1.5, 1.5, 1.5, 1.2, 85.0])


def entry_sim(p, t_end=4.5, push_ups=0, record=None):
    """Stand -> plant claws forward -> kick the rear up -> tuck -> catch the handstand
    (wrist reflex) -> optional push-ups. Returns stats."""
    m, d = setup()
    mujoco.mj_resetData(m, d)
    d.qpos[QA] = HOMEV
    mujoco.mj_forward(m, d)
    plant = vec(claws(elbow=p[0], wrist=p[1], jaw=HS["jaw"]))
    hs = vec(claws(**HS))
    kick = legs_offset(plant, *p[2:6])
    t_plant, t_kick = 0.6 + p[11], 0.6 + p[11] + p[6]
    tuck_tgt = legs_offset(hs, *p[7:11])
    low = vec(claws(elbow=HS["elbow"] + 1.3, wrist=HS["wrist"], jaw=HS["jaw"]))
    low = legs_offset(low, *p[7:11])
    dt = m.opt.timestep
    rec, hold, caught, sat, n = [], 0.0, None, 0, 0
    period = 2.4
    for k in range(int(t_end / dt)):
        t = k * dt
        nd = nose_down_deg(d)
        if t < 0.6:
            a = min(t / 0.5, 1.0)
            tgt = HOMEV + (a * a * (3 - 2 * a)) * (plant - HOMEV)
        elif t < t_plant:
            tgt = plant
        elif t < t_kick:
            tgt = kick
        else:
            tgt = tuck_tgt
            if caught is None and nd > p[12]:
                caught = t
            if caught is not None:
                if push_ups and t > caught + 1.2:
                    a = 0.5 - 0.5 * math.cos(2 * math.pi * min((t - caught - 1.2) / period, push_ups))
                    tgt = tuck_tgt + a * (low - tuck_tgt)
                tgt = reflex(tgt, d)
        tau = step(tgt, rec)
        nc, other = floor_contacts(d)
        if t > t_end - 2.0:
            n += 1
            if nd > 70 and nc > 0 and not other:
                hold += dt
        if record is not None:
            record(t, d)
    rec = np.array(rec)
    return dict(hold=hold, nose_down=nose_down_deg(d), peak=float(rec.max()),
                sat=float((rec > 0.98 * PEAK).mean()), caught=caught)


def entry_score(p) -> float:
    r = entry_sim(p)
    return r["hold"] + 0.01 * min(r["nose_down"], 90) - 2.0 * r["sat"]


def search_entry(iters=22, pop=24, elite=5, seed=3):
    import json
    import pathlib
    rng = np.random.default_rng(seed)
    mean = np.array([-0.4, 0.0, 0.3, 0.3, -0.8, -0.8, 0.18, 0.6, 0.6, 0.6, 0.6, 0.5, 60.0])
    sigma = np.array([0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.1, 0.6, 0.6, 0.6, 0.6, 0.3, 12.0])
    best, best_s = mean.copy(), -1e9
    out = pathlib.Path(__file__).with_name("handstand_entry.json")
    for it in range(iters):
        cand = np.clip(mean + sigma * rng.standard_normal((pop, len(mean))), ENTRY_LO, ENTRY_HI)
        cand[0] = best
        scores = np.array([entry_score(c) for c in cand])
        o = np.argsort(scores)[::-1]
        if scores[o[0]] > best_s:
            best_s, best = scores[o[0]], cand[o[0]].copy()
        el = cand[o[:elite]]
        mean = 0.7 * el.mean(0) + 0.3 * mean
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.02)
        r = entry_sim(best)
        print(f"iter {it:2d} best {best_s:5.2f} hold {r['hold']:.2f}/2.0 s  end nose-down {r['nose_down']:5.1f}"
              f"  peak {r['peak']:.2f}  maxed {100 * r['sat']:.1f}%  caught {r['caught']}", flush=True)
        out.write_text(json.dumps(best.tolist()))
    return best




# ── Entry v2: slow pike, then a small kick, tuck and catch ──────────────────
# params: pike claws (elbow, wrist), pike legs (m_lift, m_ext, r_lift, r_ext),
#         kick legs (4), kick time, tuck legs (4), catch angle
PIKE_LO = np.r_[[-1.2, -1.2], np.full(4, -1.6), np.full(4, -1.6), [0.05], np.full(4, -1.6), [40.0]]
PIKE_HI = np.r_[[1.6, 1.6], np.full(4, 1.6), np.full(4, 1.6), [0.40], np.full(4, 1.6), [85.0]]
T_PIKE = 1.8


def pike_sim(p, t_end=5.0, push_ups=0, record=None):
    m, d = setup()
    mujoco.mj_resetData(m, d)
    d.qpos[QA] = HOMEV
    mujoco.mj_forward(m, d)
    pike = legs_offset(vec(claws(elbow=p[0], wrist=p[1], jaw=HS["jaw"])), *p[2:6])
    kick = legs_offset(vec(claws(elbow=p[0], wrist=p[1], jaw=HS["jaw"])), *p[6:10])
    hs = legs_offset(vec(claws(**HS)), *p[11:15])
    low = legs_offset(vec(claws(elbow=HS["elbow"] + 1.3, wrist=HS["wrist"], jaw=HS["jaw"])), *p[11:15])
    t_kick = T_PIKE + 0.3
    t_tuck = t_kick + p[10]
    dt = m.opt.timestep
    rec, hold, caught, peak_nd, high = [], 0.0, None, -90.0, 0.0
    period = 2.4
    for k in range(int(t_end / dt)):
        t = k * dt
        nd = nose_down_deg(d)
        if t < T_PIKE:
            a = t / T_PIKE
            tgt = HOMEV + (a * a * (3 - 2 * a)) * (pike - HOMEV)
        elif t < t_kick:
            tgt = pike
        elif t < t_tuck:
            tgt = kick
        else:
            tgt = hs
            if len(p) > 16:                        # v3: wrists press from the moment the kick ends
                if caught is None:
                    caught = t
                if t < t_tuck + (p[18] if len(p) > 18 else 0.4):
                    tgt = tgt.copy()
                    tgt[JI["LF_J3_joint"]] += p[17]
                    tgt[JI["RF_J3_joint"]] -= p[17]
            elif caught is None and nd > p[15]:
                caught = t
            if caught is not None:
                if push_ups and t > caught + 1.2:
                    a = 0.5 - 0.5 * math.cos(2 * math.pi * min((t - caught - 1.2) / period, push_ups))
                    tgt = hs + a * (low - hs)
                tgt = reflex(tgt, d, kp=p[16] if len(p) > 16 else REFLEX_KP,
                             kd=p[19] if len(p) > 19 else 0.0)
        step(tgt, rec)
        peak_nd = max(peak_nd, nd)
        nc, other = floor_contacts(d)
        if t > t_end - 2.0 and nd > 70 and nc > 0 and not other:
            hold += dt
        if nd > 75 and nc > 0 and not other:
            high += dt
        if record is not None:
            record(t, d)
    rec = np.array(rec)
    return dict(hold=hold, nose_down=nose_down_deg(d), peak_nd=peak_nd, peak=float(rec.max()),
                sat=float((rec > 0.98 * PEAK).mean()), caught=caught, high=high)


def press2_score(p) -> float:
    r = pike_sim(p)
    return 2.0 * r["hold"] + 1.0 * r["high"] + 0.02 * min(r["peak_nd"], 90) - 2.0 * r["sat"]


def search_press2(iters=16, pop=24, elite=5, seed=9):
    import json
    import pathlib
    rng = np.random.default_rng(seed)
    prev = np.array(json.loads(pathlib.Path(__file__).with_name("handstand_press.json").read_text()))
    lo, hi = np.r_[PRESS_LO, [0.2]], np.r_[PRESS_HI, [1.2]]
    mean = np.r_[prev, [0.6]]
    mean[11:15] *= 0.3                       # legs nearer HOME, where the balanced handstand was found
    sigma = np.r_[np.full(len(prev), 0.12), [0.2]]
    sigma[10] = 0.02
    sigma[16:18] = [1.5, 0.3]
    best, best_s = mean.copy(), -1e9
    out = pathlib.Path(__file__).with_name("handstand_press2.json")
    for it in range(iters):
        cand = np.clip(mean + sigma * rng.standard_normal((pop, len(mean))), lo, hi)
        cand[0] = best
        scores = np.array([press2_score(c) for c in cand])
        o = np.argsort(scores)[::-1]
        if scores[o[0]] > best_s:
            best_s, best = scores[o[0]], cand[o[0]].copy()
        el = cand[o[:elite]]
        mean = 0.7 * el.mean(0) + 0.3 * mean
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.02)
        r = pike_sim(best)
        print(f"iter {it:2d} best {best_s:5.2f} hold {r['hold']:.2f}/2.0 s  above-75 {r['high']:.2f}s"
              f"  peak nose-down {r['peak_nd']:5.1f}  end {r['nose_down']:5.1f}  peak torque {r['peak']:.2f}"
              f"  maxed {100 * r['sat']:.1f}%", flush=True)
        out.write_text(json.dumps(best.tolist()))
    return best


def search_catch(iters=10, pop=20, elite=4, seed=11):
    """Keep the press2 entry; tune only the catch: reflex kp, kd, press size and duration."""
    import json
    import pathlib
    rng = np.random.default_rng(seed)
    prev = np.array(json.loads(pathlib.Path(__file__).with_name("handstand_press2.json").read_text()))
    base = np.r_[prev, [0.0]]
    free = [16, 17, 18, 19]
    lo, hi = np.array([1.0, -1.6, 0.1, -0.4]), np.array([12.0, 1.6, 1.2, 0.4])
    mean, sigma = base[free].copy(), np.array([2.0, 0.3, 0.2, 0.15])
    best, best_s = base.copy(), -1e9
    out = pathlib.Path(__file__).with_name("handstand_catch.json")
    for it in range(iters):
        cands = []
        for _ in range(pop):
            c = base.copy()
            c[free] = np.clip(mean + sigma * rng.standard_normal(4), lo, hi)
            cands.append(c)
        cands[0] = best
        scores = np.array([press2_score(c) for c in cands])
        o = np.argsort(scores)[::-1]
        if scores[o[0]] > best_s:
            best_s, best = scores[o[0]], cands[o[0]].copy()
        el = np.array([cands[i][free] for i in o[:elite]])
        mean = 0.7 * el.mean(0) + 0.3 * mean
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.01)
        r = pike_sim(best)
        print(f"iter {it:2d} best {best_s:5.2f} hold {r['hold']:.2f}/2.0 s  above-75 {r['high']:.2f}s"
              f"  peak nose-down {r['peak_nd']:5.1f}  end {r['nose_down']:5.1f}  maxed {100 * r['sat']:.1f}%"
              f"  kp {best[16]:.1f} kd {best[19]:+.2f}", flush=True)
        out.write_text(json.dumps(best.tolist()))


def pike_score(p) -> float:
    r = pike_sim(p)
    return 2.0 * r["hold"] + 0.02 * min(r["peak_nd"], 90) + 0.01 * min(r["nose_down"], 90) - 2.0 * r["sat"]


PRESS_LO = np.r_[PIKE_LO, [1.0, -1.6]]
PRESS_HI = np.r_[PIKE_HI, [10.0, 1.6]]


def search_press(iters=22, pop=24, elite=5, seed=7):
    """v3: warm start from the pike result; add wrist press gain + feedforward."""
    import json
    import pathlib
    rng = np.random.default_rng(seed)
    pike = np.array(json.loads(pathlib.Path(__file__).with_name("handstand_pike.json").read_text()))
    mean = np.r_[pike, [4.5, 0.0]]
    sigma = np.r_[np.full(len(pike), 0.15), [2.5, 0.6]]
    sigma[10] = 0.03
    best, best_s = mean.copy(), -1e9
    out = pathlib.Path(__file__).with_name("handstand_press.json")
    for it in range(iters):
        cand = np.clip(mean + sigma * rng.standard_normal((pop, len(mean))), PRESS_LO, PRESS_HI)
        cand[0] = best
        scores = np.array([pike_score(c) for c in cand])
        o = np.argsort(scores)[::-1]
        if scores[o[0]] > best_s:
            best_s, best = scores[o[0]], cand[o[0]].copy()
        el = cand[o[:elite]]
        mean = 0.7 * el.mean(0) + 0.3 * mean
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.02)
        r = pike_sim(best)
        print(f"iter {it:2d} best {best_s:5.2f} hold {r['hold']:.2f}/2.0 s  peak nose-down {r['peak_nd']:5.1f}"
              f"  end {r['nose_down']:5.1f}  peak torque {r['peak']:.2f}  maxed {100 * r['sat']:.1f}%", flush=True)
        out.write_text(json.dumps(best.tolist()))
    return best


def search_pike(iters=26, pop=24, elite=5, seed=5):
    import json
    import pathlib
    rng = np.random.default_rng(seed)
    mean = np.r_[[0.3, 0.1], [-0.3, -0.3, -0.8, -0.8], [-0.6, -0.6, -1.2, -1.2], [0.15],
                 [0.8, 0.8, 0.8, 0.8], [65.0]]
    sigma = np.r_[[0.5, 0.5], np.full(4, 0.6), np.full(4, 0.6), [0.08], np.full(4, 0.6), [10.0]]
    best, best_s = mean.copy(), -1e9
    out = pathlib.Path(__file__).with_name("handstand_pike.json")
    for it in range(iters):
        cand = np.clip(mean + sigma * rng.standard_normal((pop, len(mean))), PIKE_LO, PIKE_HI)
        cand[0] = best
        scores = np.array([pike_score(c) for c in cand])
        o = np.argsort(scores)[::-1]
        if scores[o[0]] > best_s:
            best_s, best = scores[o[0]], cand[o[0]].copy()
        el = cand[o[:elite]]
        mean = 0.7 * el.mean(0) + 0.3 * mean
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.02)
        r = pike_sim(best)
        print(f"iter {it:2d} best {best_s:5.2f} hold {r['hold']:.2f}/2.0 s  peak nose-down {r['peak_nd']:5.1f}"
              f"  end {r['nose_down']:5.1f}  peak torque {r['peak']:.2f}  maxed {100 * r['sat']:.1f}%"
              f"  caught {r['caught']}", flush=True)
        out.write_text(json.dumps(best.tolist()))
    return best


if __name__ == "__main__":
    import sys
    {"entry": search_entry, "pike": search_pike, "press": search_press, "press2": search_press2, "catch": search_catch}[sys.argv[1] if len(sys.argv) > 1 else "pike"]()
