"""Reference 4: the flat-crab cartwheel with a running start and a sideways jump.

Like reference 3, but the slow tip onto the edge is replaced by a jump: crouch,
then push off with the left legs driving harder, so the body is lifted to wheel
height and tips onto its right edge in the air, snaps into the six-spoke star
(~0.1 s, measured), and plants a single spoke (right middle leg) ~17 deg ahead
of the hub -- a running jump onto one spoke. The hub then vaults over it like a
pole vaulter, which turns the run-up's forward speed into wheel spin, and rolls
on. It lands leaning -- the star tipped PHI_TD of the way to its edge, which
sits the hub lower, nearer the height the jump reaches -- and stands up fully on
edge over the first RISE degrees of the roll. The jump supplies the ~4-5 J of lift that, in reference 3, used up the
run-up's momentum. Take-off speed 1.55 m/s up, as the robot's own recorded
jump reaches.

(Reference 3 notes follow.)

  0.0 - 2.0 s  run-up: a physics recording of the repo's trained walking policy
               (jumper.posture, tripod gait) trotting straight ahead
               (record_runup.py -> runup_walk.npz)
  then         tip onto the right-hand edge while still moving forward, limbs
               spreading into the flat six-spoke star; the wheel starts turning at
               the speed the run-up gives it (forward speed / wheel radius)
  then         roll one full turn at an even pace (ramping to ROLL_RATE), the
               shell never on the floor
  then         tip back down onto its feet and stand

Same npz format as references 1 and 2. The run-up is physical; the rest is
kinematic (lowest point on the floor, no slip at the pivot tip), for the RL policy
to make physical.

  python cartwheel_ref4.py   # writes tasks/jumper/cartwheel_jump/ref/cartwheel_jump.npz
"""
from __future__ import annotations

import json
import math
import pathlib

import mujoco
import numpy as np

from tools.cartwheel.refs.cartwheel_ref import CHASSIS, FEET, LEGS, TIPS, mj
from tools.cartwheel.refs.cartwheel_ref2 import flat_star, quat_mul
from tools.crab_trio.stage import HOME, JOINTS

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
XML = ROOT / "assets/jumper/jumper.xml"
OUT = ROOT / "tasks/jumper/cartwheel_jump/ref/cartwheel_jump.npz"

REC_HZ, CONTROL_HZ, T_GO = 250.0, 200.0, 0.3
T_TIP = 0.45          # (reference 3's slow tip; unused here)
T_JUMP = 0.40         # crouch 0.12 s, push 0.08 s, then airborne into the star
T_TAKEOFF = 0.20
VZ_TAKEOFF = 1.55     # m/s, the robot's recorded jump
R_LAND = 0.27
THETA_TD = -0.30     # rad: wheel angle at touchdown -- the landing spoke planted ~17 deg AHEAD of the hub
R_SPOKE = 0.31       # m, hub height over a vertical spoke (vault radius)
PHI_TD = 55.0        # deg: how far the star is tipped toward its edge at touchdown (90 = fully on edge)
RISE = 150.0         # deg of wheel turn over which it stands up from PHI_TD to fully on edge         # hub height landing on two spokes 30 deg either side of vertical
ROLL_RATE = 3.2       # rad/s, even rolling pace
RAMP = 0.4            # s to go from the run-up's spin to ROLL_RATE
T_DOWN, T_STAND = 0.5, 0.4
TURNS = 1.0           # full turns of the wheel in the roll segment (cartwheel_ref5.py sets more)
#: optional heading profile for the roll: psi(s) in rad, s = 0..1 through the roll's
#: turns; the whole motion is yawed by psi about the hub from the roll on (a figure 8:
#: 0 -> 2 pi -> 0). None = straight.
HEADING_FN = None


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_runup(joint_order):
    r = np.load(HERE / "runup_walk.npz")
    names = [str(n) for n in r["joint_names"]]
    perm = [names.index(j) for j in joint_order]
    t = r["t"]
    pos, quat, q = r["base_pos"].copy(), r["base_quat"].copy(), r["q"][:, perm]
    # express in a frame where the walk starts at the origin heading along +x
    yaw0 = yaw_of(quat[0])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos[:, :2] -= pos[0, :2]
    pos = pos @ rot.T
    qz = np.array([math.cos(-yaw0 / 2), 0, 0, math.sin(-yaw0 / 2)])
    quat = np.array([quat_mul(qz, qq) for qq in quat])
    return t, pos, quat, q


def build():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
    lo = np.array([m.jnt_range[m.joint(j).id][0] for j in JOINTS])
    hi = np.array([m.jnt_range[m.joint(j).id][1] for j in JOINTS])
    base = m.body("base_link").id
    tips = [m.body(b).id for b in TIPS]
    feet = [m.body(b).id for b in FEET]
    chassis = []
    for g in range(m.ngeom):
        if (m.body(m.geom_bodyid[g]).name in CHASSIS and m.geom_contype[g]
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            chassis.append((g, m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]))
    home = np.array([HOME[j] for j in JOINTS])
    flat = np.clip(flat_star(), lo, hi)

    def place(qj, quat, x, y, z=0.0):
        d.qpos[:] = 0
        d.qpos[0:3] = [x, y, z]
        d.qpos[3:7] = quat
        d.qpos[qa] = qj
        mujoco.mj_kinematics(m, d)

    def chassis_min_z():
        return min(float(((d.geom_xmat[g].reshape(3, 3) @ v.T)[2] + d.geom_xpos[g][2]).min())
                   for g, v in chassis)

    place(home, [1, 0, 0, 0], 0, 0)
    r_foot = 0.10647 + min(d.xpos[b][2] for b in tips)

    # ---- run-up (physics recording) resampled to REC_HZ, with the stand before go
    rt, rpos, rquat, rq = load_runup(JOINTS)
    T_RUN = float(rt[-1])
    v_run = float((rpos[-1, 0] - rpos[-int(0.5 / (rt[1] - rt[0])), 0]) / 0.5)
    yaw_end = yaw_of(rquat[-1])

    frames = []          # (q, base_pos, base_quat, phase-label)
    for ti in np.arange(0, T_GO, 1 / REC_HZ):
        frames.append((rq[0], rpos[0], rquat[0], "stand"))
    for ti in np.arange(0, T_RUN, 1 / REC_HZ):
        k = min(int(ti / (rt[1] - rt[0])), len(rt) - 2)
        a = (ti - rt[k]) / (rt[1] - rt[0])
        qq = rquat[k] * (1 - a) + rquat[k + 1] * a
        frames.append((rq[k] * (1 - a) + rq[k + 1] * a, rpos[k] * (1 - a) + rpos[k + 1] * a,
                       qq / np.linalg.norm(qq), "run-up"))

    # ---- tip on, roll, tip down: kinematic, continuing from the run-up's last state
    q_run_end = np.clip(rq[-1], lo, hi)
    x, y = float(rpos[-1, 0]), float(rpos[-1, 1])
    heading = np.array([math.cos(yaw_end / 2), 0, 0, math.sin(yaw_end / 2)])
    r_wheel = 0.30
    w_run = v_run / r_wheel
    theta = 0.0
    pivot = pivot_xy = None
    t = 0.0
    dt = 1 / REC_HZ
    chassis_touch = 0
    stage, t_stage = "jump", 0.0
    stance_v = np.clip(np.array([HOME[j] for j in JOINTS]), lo, hi)
    crouch_v = stance_v.copy()
    push_v = stance_v.copy()
    for i, j in enumerate(JOINTS):
        if j[1] in "MR" and j[3:5] in ("J1", "J2"):
            sgn = 1.0 if j[0] == "L" else -1.0
            crouch_v[i] += sgn * 0.35                                  # fold under the body
            push_v[i] -= sgn * (0.55 if j[0] == "L" else 0.25)         # left side drives harder
    crouch_v, push_v = np.clip(crouch_v, lo, hi), np.clip(push_v, lo, hi)
    z_take = None
    BPz_prev = 0.0
    thetas = []
    while stage != "done":
        if stage == "jump":
            s = t_stage / T_JUMP
            if t_stage < 0.12:
                qj = q_run_end + mj(t_stage / 0.12) * (crouch_v - q_run_end)
            elif t_stage < T_TAKEOFF:
                qj = crouch_v + mj((t_stage - 0.12) / 0.08) * (push_v - crouch_v)
            else:
                qj = push_v + mj((t_stage - T_TAKEOFF) / 0.12) * (flat - push_v)
            phi = math.radians(PHI_TD) * mj(s)
            # in the air: tip onto the right edge and turn the wheel back a little so the
            # right-middle spoke reaches the floor ahead of the hub (a pole-vault plant)
            theta = THETA_TD * mj((t_stage - T_TAKEOFF) / (T_JUMP - T_TAKEOFF))
            wdot = 0.0
            slide = v_run                           # airborne: forward speed carries on
            if t_stage >= T_JUMP:
                stage, t_stage = "roll", 0.0
                # touchdown on one spoke: the speed across the spoke becomes the vault's spin
                w_run = v_run * math.cos(THETA_TD) / R_SPOKE
                continue
        elif stage == "tip-on":
            s = t_stage / T_TIP
            qj = q_run_end + mj(s) * (flat - q_run_end)
            phi = math.radians(90) * mj(s)
            wdot = w_run * s                       # the wheel picks up the run-up's speed
            slide = v_run * (1 - mj(s))            # forward momentum carried while not yet rolling
            if t_stage >= T_TIP:
                stage, t_stage = "roll", 0.0
                continue
        elif stage == "roll":
            qj = flat
            # lands leaning (wheel plane PHI_TD from flat), stands fully on edge while rolling
            rise = mj((theta - THETA_TD) / math.radians(RISE))
            phi = math.radians(PHI_TD + (90.0 - PHI_TD) * rise)
            wdot = w_run + (ROLL_RATE - w_run) * mj(t_stage / RAMP)
            slide = 0.0
            if theta >= 2 * math.pi * TURNS:
                theta = 2 * math.pi * TURNS
                stage, t_stage = "tip-down", 0.0
                continue
        elif stage == "tip-down":
            s = t_stage / T_DOWN
            qj = flat + mj(s / 0.75) * (home - flat)
            phi, wdot, slide = math.radians(90) * (1 - mj(s)), 0.0, 0.0
            if t_stage >= T_DOWN:
                stage, t_stage = "stand-end", 0.0
                continue
        else:
            qj, phi, wdot, slide = home, 0.0, 0.0, 0.0
            if t_stage >= T_STAND:
                stage = "done"
                continue
        if stage != "jump":
            theta = min(theta + wdot * dt, 2 * math.pi * TURNS)
        quat = quat_mul(heading, quat_mul([math.cos(phi / 2), math.sin(phi / 2), 0, 0],
                                          [math.cos(theta / 2), 0, 0, -math.sin(theta / 2)]))  # rolls forward
        qj = np.clip(qj, lo, hi)
        x += slide * math.cos(yaw_end) * dt
        y += slide * math.sin(yaw_end) * dt
        place(qj, quat, x, y)
        tz = np.array([d.xpos[b][2] for b in tips])
        k = int(np.argmin(tz))
        if k == pivot and slide == 0.0:
            x += pivot_xy[0] - d.xpos[tips[k]][0]
            y += pivot_xy[1] - d.xpos[tips[k]][1]
            place(qj, quat, x, y)
        pivot, pivot_xy = k, d.xpos[tips[k]][:2].copy()
        z_tips = r_foot - tz.min()
        cz = chassis_min_z()
        z = max(z_tips, -cz)
        if stage == "jump" and t_stage >= T_TAKEOFF:   # ballistic flight from take-off
            if z_take is None:
                z_take = float(BPz_prev)
            ta = t_stage - T_TAKEOFF
            z = max(z, z_take + VZ_TAKEOFF * ta - 0.5 * 9.81 * ta * ta)
        BPz_prev = z
        if -cz > z_tips + 1e-4 and stage in ("roll",):
            chassis_touch += 1
        place(qj, quat, x, y, z)
        frames.append((qj, d.xpos[base].copy(), np.array(quat), stage))
        thetas.append(theta)
        t += dt
        t_stage += dt

    n = len(frames)
    tt = np.arange(n) / REC_HZ
    Q = np.array([f[0] for f in frames])
    BP = np.array([f[1] for f in frames])
    BQ = np.array([f[2] for f in frames])
    labels = [f[3] for f in frames]
    if HEADING_FN is not None:              # yaw the motion about the hub from the roll on
        i_r = labels.index("roll")
        th = np.concatenate([np.zeros(i_r), np.array(thetas)])[:n]
        span = 2 * math.pi * TURNS - THETA_TD
        psi = np.array([HEADING_FN(min(max((th[i] - THETA_TD) / span, 0.0), 1.0)) if i >= i_r else 0.0
                        for i in range(n)])
        BP2 = BP.copy()
        for i in range(i_r, n):
            c, s_ = math.cos(psi[i]), math.sin(psi[i])
            dxy = BP[i, :2] - BP[i - 1, :2]
            BP2[i, :2] = BP2[i - 1, :2] + [c * dxy[0] - s_ * dxy[1], s_ * dxy[0] + c * dxy[1]]
            BQ[i] = quat_mul([math.cos(psi[i] / 2), 0, 0, math.sin(psi[i] / 2)], BQ[i])
        BP = BP2
    # smooth the hand-off from the recorded run-up into the kinematic part (base height)
    i0 = labels.index("jump")
    blend = int(0.15 * REC_HZ)
    dz = BP[i0 - 1, 2] - BP[i0, 2]
    for j in range(blend):
        BP[i0 + j, 2] += dz * (1 - j / blend)
    FP = np.zeros((n, 6, 3))
    for i in range(n):
        place(Q[i], BQ[i], *BP[i])
        FP[i] = [d.xpos[b] for b in feet]
    AV = np.zeros((n, 3))
    for i in range(1, n):
        dq = quat_mul(BQ[i], quat_conj(BQ[i - 1]))
        if dq[0] < 0:
            dq = -dq
        ang = 2 * math.atan2(np.linalg.norm(dq[1:]), dq[0])
        AV[i] = dq[1:] / (np.linalg.norm(dq[1:]) + 1e-12) * ang * REC_HZ
    QD = np.gradient(Q, dt, axis=0)
    LV = np.gradient(BP, dt, axis=0)
    # smooth the velocity channel (the kinematic hand-offs at take-off and landing
    # leave one-frame glitches in the finite differences)
    k = 25
    ker = np.ones(k) / k
    LV = np.stack([np.convolve(np.pad(LV[:, a], k // 2, mode="edge"), ker, mode="valid") for a in range(3)], 1)
    contact = (FP[:, :, 2] - FP[:, :, 2].min(0, keepdims=True) < 0.010)
    qc_t = np.arange(int(round(tt[-1] * CONTROL_HZ)) + 1) / CONTROL_HZ
    QC = np.stack([np.interp(qc_t, tt, Q[:, j]) for j in range(len(JOINTS))], 1)
    seg = {s: [labels.index(s) / REC_HZ - T_GO, (len(labels) - labels[::-1].index(s)) / REC_HZ - T_GO]
           for s in ("run-up", "jump", "roll", "tip-down", "stand-end")}
    roll_mask = np.array([lab == "roll" for lab in labels])
    meta = dict(
        description="flat-crab cartwheel with a running start and a sideways jump: recorded "
                    "tripod trot -> crouch, sideways jump onto the edge, star in the air -> "
                    "360-degree roll on the six-spoke star -> stand (tools/cartwheel/refs/cartwheel_ref4.py)",
        joint_order=list(JOINTS), leg_order=list(LEGS), rec_hz=REC_HZ, control_hz=CONTROL_HZ,
        t_go=T_GO, go_frame=int(round(T_GO * REC_HZ)), stand_base_z=float(BP[0, 2]),
        kp=20.0, kd=0.5, tau_limit=1.7464, kinematic="run-up recorded, rest kinematic",
        segments_s_after_go={k: [round(a, 3), round(b, 3)] for k, (a, b) in seg.items()},
        stats=dict(runup_speed=v_run, roll_rate=ROLL_RATE, turns=TURNS,
                   heading=None if HEADING_FN is None else getattr(HEADING_FN, "__doc__", "custom"),
                   rim_speed_while_rolling=float(np.linalg.norm(LV[roll_mask, :2], axis=1).mean()),
                   total_travel=float(BP[-1, 0] - BP[0, 0]),
                   path_length=float(np.linalg.norm(np.diff(BP[:, :2], axis=0), axis=1).sum()),
                   base_z_range_while_rolling=[float(BP[roll_mask, 2].min()), float(BP[roll_mask, 2].max())],
                   chassis_on_floor_s=chassis_touch / REC_HZ, duration=float(tt[-1])),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, meta=json.dumps(meta), t=tt, q=Q, qd=QD, q_cmd=QC, base_pos=BP, base_quat=BQ,
             base_lin_vel=LV, base_ang_vel=AV, foot_pos=FP, contact=contact)
    print("wrote", OUT)
    print(json.dumps({"segments": meta["segments_s_after_go"], **meta["stats"]}, indent=1))


if __name__ == "__main__":
    build()
