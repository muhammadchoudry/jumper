"""Limb-over-limb cartwheel: physics check of a scripted controller (no learning).

A rigid six-spoke star loses ~2/3 of its spin energy at every spoke change (the
next spoke lands 60 deg from the last) and must lift its hub ~4 cm over each
spoke -- it cannot coast a full turn on any run-up the crab can make (checked
below: the rigid star stops after ~25 deg from 2.4 rad/s). Here each leg works
instead, like a hand in a human cartwheel:

  swing    in the air the leg shortens (folds), swings forward, straightens
  plant    it lands ~STEP/2 ahead of straight-down
  carry    while it carries the body the hip follows the step schedule and the
           leg straightens from slightly bent (pumping, like on a swing: the
           only way motors add energy with one foot down -- a hip torque alone
           just trades body spin against travel around the foot)
  release  when the next leg lands it lets go: it holds still in the body and
           shortens, so the body turning over the new leg lifts it off

so consecutive legs meet the floor ~STEP apart (not 60 deg).

Each foot is placed in the wheel's own frame -- angle around the hub, length,
offset out of the wheel's plane -- and inverse kinematics sets the leg's joints
(the hip-yaw axis is tilted ~40 deg from the axle, so J0 alone also tilts the
leg out of the plane and shortens it).

Balance: a thin wheel at ~1 m/s gets no help from its spin, so it steps under
itself: legs about to land, and the one carrying the body, are placed out of the
wheel's plane toward the side it is leaning to.

  python limb_wheel.py hand               # rolling only, sideways balance held by an invisible hand
  python limb_wheel.py search_hand N      # search the rolling gains with the hand
  python limb_wheel.py search N           # search everything without the hand
  python limb_wheel.py best               # replay the best found
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
from cartwheel_ref import TIPS, mj
from cartwheel_ref2 import flat_star, quat_mul
from stage import JOINTS

HERE = pathlib.Path(__file__).resolve().parent
PARAMS = HERE / "limb_wheel_params.json"
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")          # order of TIPS
STAR = flat_star()
LEG_J = [[JOINTS.index(f"{L}_{j}_joint") for j in (("J0", "J1", "J2", "J3") if L[1] == "F" else ("J0", "J1", "J2"))]
         for L in LEGS]
CTRL_EVERY = 5                                        # control at 200 Hz on the 1 kHz sim

# params: step (deg between consecutive plants), s_land (leg length at touch-down, fraction
# of straight), x_full (deg past down where it is straight again), lift (extra fold in the
# air), k (m of sideways foot offset per rad of lean), kd (s), win_lo/win_hi (deg of rest
# angle ahead of down where legs place for balance)
DEFAULT = dict(step=30.0, s_land=0.87, x_full=-20.0, lift=0.12, k=0.15, kd=0.15, win_lo=-30.0, win_hi=75.0)
KEYS = list(DEFAULT)


class Kin:
    """Kinematic model of the crab with its body fixed at the origin (the body frame)."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(HERE.parents[1] / "assets/jumper/jumper.xml"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self.dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
        self.tips = [m.body(b).id for b in TIPS]
        self.lo = np.array([m.jnt_range[m.joint(j).id][0] for j in JOINTS])
        self.hi = np.array([m.jnt_range[m.joint(j).id][1] for j in JOINTS])
        self.jp = np.zeros((3, m.nv))

    def tips_at(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[3] = 1
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return np.array([self.d.xpos[t] for t in self.tips])

    def ik(self, q, targets, iters=3, lam=0.02):
        """Damped least squares, all six feet at once, warm-started from q."""
        q = q.copy()
        for _ in range(iters):
            p = self.tips_at(q)
            mujoco.mj_comPos(self.m, self.d)
            for i in range(6):
                e = targets[i] - p[i]
                mujoco.mj_jacBody(self.m, self.d, self.jp, None, self.tips[i])
                J = self.jp[:, self.dof[LEG_J[i]]]
                dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(3), e)
                q[LEG_J[i]] = np.clip(q[LEG_J[i]] + dq, self.lo[LEG_J[i]], self.hi[LEG_J[i]])
        return q


KIN = Kin()
P0 = KIN.tips_at(STAR)                                # star feet in the body frame
ANG = np.arctan2(P0[:, 1], P0[:, 0])
RAD = np.hypot(P0[:, 0], P0[:, 1])
HIP = np.array([KIN.d.xanchor[KIN.m.joint(f"{L}_J0_joint").id] for L in LEGS])
LEGV = P0 - HIP                                       # hip -> foot, straight leg in the star
# per-leg length that puts every foot at the shortest leg's reach from the hub (a round
# wheel): the claws are 2.8 cm and the rear legs 1.8 cm longer than the middle legs
EQ = 1.0 - (RAD - RAD.min()) / np.linalg.norm(LEGV, axis=1)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def nominal_delta(x: float, half: float) -> float:
    """Foot angle offset (deg, + = turned back) for a leg whose rest angle is x deg ahead of down."""
    if -30.0 <= x <= 30.0:                    # carrying: from +half ahead to -half behind
        return -x * half / 30.0
    u = ((-30.0 - x) % 360.0) / 300.0         # in the air: swing forward again
    return half - 2 * half * mj(u)


def swing_u(x: float) -> float:
    return ((-30.0 - x) % 360.0) / 300.0


def foot(i, delta_deg, scale, zoff):
    """Foot target: the star leg turned delta about its own hip (in the wheel plane),
    scaled in length, offset out of the plane -- a straight leg stays straight."""
    a = math.radians(delta_deg)
    c, s_ = math.cos(a), math.sin(a)
    v = LEGV[i] * scale * EQ[i]
    return HIP[i] + np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2] + zoff])


def foot_delta(i, f) -> float:
    """Angle (deg) of a foot about its hip, relative to the star leg."""
    v = f - HIP[i]
    return wrap180(math.degrees(math.atan2(v[1], v[0]) - math.atan2(LEGV[i, 1], LEGV[i, 0])))


def balance_offset(x, lean, lean_rate, p):
    if p["win_lo"] <= x <= p["win_hi"]:
        return -p["k"] * (lean + p["kd"] * lean_rate)
    return 0.0


class Controller:
    """Per leg: air -> push (on touch-down, front region) -> release (when the next leg
    ahead lands) -> air. Only the front-most planted leg pushes."""

    def __init__(self, p: dict):
        self.p = p
        self.state = ["air"] * 6
        self.hold = np.zeros(6)
        self.t_rel = np.zeros(6)
        self.q = STAR.copy()

    def __call__(self, t, theta, lean, lean_rate, feet_body, tipz):
        p, half = self.p, self.p["step"] / 2.0
        xs = [wrap180(math.degrees(ANG[i]) + 90.0 - math.degrees(theta)) for i in range(6)]
        dact = [foot_delta(i, f) for i, f in enumerate(feet_body)]
        for i in range(6):
            down = tipz[i] < 0.0165
            if self.state[i] == "air" and down and -40.0 <= xs[i] <= 70.0:
                self.state[i] = "push"
                for k in range(6):                # the leg it replaces lets go
                    if k != i and self.state[k] == "push" and xs[k] < xs[i]:
                        self.state[k], self.hold[k], self.t_rel[k] = "release", dact[k], t
            elif self.state[i] == "release" and (tipz[i] > 0.03 or xs[i] < -90.0):
                self.state[i] = "air"
            elif self.state[i] == "push" and xs[i] < -90.0:
                self.state[i] = "air"
        targets = np.zeros((6, 3))
        for i in range(6):
            z = balance_offset(xs[i], lean, lean_rate, p)
            if self.state[i] == "push":
                # carrying: the hip follows the step schedule; the leg straightens from
                # s_land to full length as the body rolls over it (pumping, like a swing)
                a = mj((30.0 - xs[i]) / (30.0 - p["x_full"]))
                targets[i] = foot(i, nominal_delta(xs[i], half), p["s_land"] + (1 - p["s_land"]) * a, z)
            elif self.state[i] == "release":
                # unloaded by the new leg: hold still in the body and fold, so it lifts off
                s = 1.0 - (1.0 - p["s_land"] + p["lift"]) * min(1.0, (t - self.t_rel[i]) / 0.05)
                targets[i] = foot(i, self.hold[i], s, z)
            else:
                u = 1.0 if -30.0 <= xs[i] <= 30.0 else swing_u(xs[i])
                s = p["s_land"] - p["lift"] * math.sin(math.pi * u / 0.85) if u < 0.85 else p["s_land"]
                targets[i] = foot(i, nominal_delta(xs[i], half), s, z)
        self.q = KIN.ik(self.q, targets)
        return self.q


def start_state(m, d, w0, theta0=-0.26):
    mujoco.mj_resetData(m, d)
    q = quat_mul([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
                 [math.cos(theta0 / 2), 0, 0, -math.sin(theta0 / 2)])
    d.qpos[3:7] = q
    d.qpos[W.QA] = STAR
    mujoco.mj_kinematics(m, d)
    tz = [d.xpos[b][2] for b in W.TIPB]
    k = int(np.argmin(tz))
    d.qpos[2] += -tz[k] + 0.0105
    mujoco.mj_kinematics(m, d)
    R = d.xmat[W.BASE].reshape(3, 3)
    wvec = -w0 * R[:, 2]                                          # forward roll about the axle
    d.qvel[0:3] = np.cross(wvec, d.xpos[W.BASE] - d.xpos[W.TIPB[k]])   # pivot on the low tip
    d.qvel[3:6] = [0, 0, -w0]
    mujoco.mj_forward(m, d)


def simulate(p: dict, w0=2.4, t_end=4.0, hand=False, rigid=False, frame_cb=None, log=None):
    m, d = W._setup()
    start_state(m, d, w0)
    dt = m.opt.timestep
    theta, prev = -0.26, None
    ctl = Controller(p)
    tgt = STAR.copy()
    lean_prev = 0.0
    hand_sq = 0.0
    max_lean = bad_t = 0.0
    spins, n = [], 0
    sat = steps = 0
    for step in range(int(t_end / dt)):
        t = step * dt
        R = d.xmat[W.BASE].reshape(3, 3)
        bx, axle = R[:, 0], R[:, 2]
        ang = math.atan2(-bx[2], bx[0])
        if prev is not None:
            theta += (ang - prev + math.pi) % (2 * math.pi) - math.pi
        prev = ang
        lean = math.asin(max(-1.0, min(1.0, axle[2])))           # 0 = on edge
        lean_rate = (lean - lean_prev) / dt
        lean_prev = lean
        if step % CTRL_EVERY == 0 and not rigid:
            feet_body = [(d.xpos[b] - d.xpos[W.BASE]) @ R for b in W.TIPB]
            tipz = [d.xpos[b][2] for b in W.TIPB]
            tgt = ctl(t, theta, lean, lean_rate, feet_body, tipz)
            if log is not None:
                log.append((round(t, 3), round(math.degrees(theta), 1), round(-float(d.qvel[5]), 2),
                            round(float(d.xpos[W.BASE][2]), 3), "".join(s[0] for s in ctl.state),
                            "".join("*" if z < 0.0165 else "." for z in tipz)))
        tau = jump.torque(tgt, d.qpos[W.QA], d.qvel[W.VA])
        d.ctrl[W.AI] = tau
        sat += int(np.sum(np.abs(tau) > 0.98 * 1.7464))
        steps += 1
        d.xfrc_applied[W.BASE] = 0
        if hand:                      # invisible hand: holds the wheel upright and on heading only
            w = R @ d.qvel[3:6]                                         # world angular velocity
            yaw = math.atan2(bx[1], bx[0]) if abs(bx[2]) < 0.9 else 0.0
            tq = np.array([300.0 * lean * np.sign(-axle[1]) - 15.0 * w[0], 0.0, -60.0 * yaw - 6.0 * w[2]])
            d.xfrc_applied[W.BASE, 3:6] = tq
            hand_sq += tq[0] ** 2 + tq[2] ** 2
        mujoco.mj_step(m, d)
        max_lean = max(max_lean, abs(lean))
        for c in d.contact[:d.ncon]:
            g = c.geom1 if c.geom2 == W.FLOOR else c.geom2 if c.geom1 == W.FLOOR else -1
            if g >= 0 and W.BADG[g]:
                bad_t += dt
                break
        if step % 50 == 0:
            spins.append(-float(d.qvel[5]))
        if frame_cb:
            frame_cb(t, d)
        if d.xpos[W.BASE][2] < 0.13 or abs(lean) > 1.2:
            break
        n += 1
    return dict(turns=theta / (2 * math.pi), t=n * dt, max_lean=math.degrees(max_lean), shell_s=bad_t,
                hand_rms=math.sqrt(hand_sq / max(n, 1)), sat=sat / max(steps * 22, 1),
                spin=[round(s, 1) for s in spins[::4]], travel=float(d.xpos[W.BASE][0]))


def show(r):
    return (f"turns {r['turns']:+.2f} in {r['t']:.2f}s  travel {r['travel']:.2f}m  max lean {r['max_lean']:.0f}deg"
            f"  shell {r['shell_s']:.2f}s  hand {r['hand_rms']:.2f}Nm  motors-maxed {100 * r['sat']:.0f}%"
            f"\n      spin every 0.2s: {r['spin']}")


def score(v, hand=False):
    p = dict(zip(KEYS, v))
    r = simulate(p, hand=hand)
    return min(r["turns"], 2.0) + 0.3 * min(r["t"], 4.0) - 2.0 * r["shell_s"]


def score_hand(v):
    return score(v, hand=True)


LO = np.array([16.0, 0.75, -30.0, 0.0, -0.5, 0.0, -60.0, 20.0])
HI = np.array([45.0, 1.0, 10.0, 0.25, 0.5, 0.4, 0.0, 120.0])


def search(iters, hand, free):
    rng = np.random.default_rng(3)
    mean = np.array([DEFAULT[k] for k in KEYS])
    if PARAMS.exists():
        mean = np.array([json.loads(PARAMS.read_text())[k] for k in KEYS])
    sigma = np.where(free, (HI - LO) / 4, 0.0)
    best, best_s = mean.copy(), -1e9
    fn = score_hand if hand else score
    with Pool(2) as pool:
        for it in range(iters):
            cand = np.clip(mean + sigma * rng.standard_normal((32, len(KEYS))), LO, HI)
            cand[0] = best
            s = np.array(pool.map(fn, list(cand)))
            o = np.argsort(s)[::-1]
            if s[o[0]] > best_s:
                best_s, best = s[o[0]], cand[o[0]].copy()
            el = cand[o[:6]]
            mean = 0.7 * el.mean(0) + 0.3 * mean
            sigma = np.where(free, np.maximum(0.7 * el.std(0) + 0.3 * sigma, (HI - LO) / 60), 0.0)
            PARAMS.write_text(json.dumps(dict(zip(KEYS, best.tolist()))))
            print(f"iter {it} best {best_s:.2f} {np.round(best, 2)}\n  ",
                  show(simulate(dict(zip(KEYS, best)), hand=hand)), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "hand":
        print("rigid star, hand:  ", show(simulate(DEFAULT, hand=True, rigid=True)))
        for step in (24.0, 32.0, 40.0):
            for s_land in (0.85, 0.92, 1.0):
                p = dict(DEFAULT, step=step, s_land=s_land)
                print(f"step {step:.0f} land at {s_land:.2f} length, hand:", show(simulate(p, hand=True)))
    elif cmd == "search_hand":         # rolling gains only (balance held by the hand)
        free = np.array([k in ("step", "s_land", "x_full", "lift") for k in KEYS])
        search(int(sys.argv[2]), True, free)
    elif cmd == "search":              # everything, no hand
        search(int(sys.argv[2]), False, np.ones(len(KEYS), bool))
    elif cmd == "best":
        p = json.loads(PARAMS.read_text())
        print("hand:   ", show(simulate(p, hand=True)))
        print("no hand:", show(simulate(p)))
