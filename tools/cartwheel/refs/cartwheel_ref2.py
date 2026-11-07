"""Reference 2: the flat-crab cartwheel.

All six limbs straight and spread in the plane of the body -- a six-spoke star,
claws front-left/front-right, middle legs out to the sides, rear legs back-left/
back-right, 60 degrees apart (hip yaw aims each spoke). The crab tips up onto its
right-hand edge (shell facing sideways), rolls 360 degrees head-over-tail along
the ground like a wheel on those spokes, then tips back down onto its feet.

In this pose the wheel's axle is the body's vertical axis, which is the hip-yaw
axis, and a straight spoke loaded along its length passes through its hip and
knee axes -- so the load is carried by the limbs, not the motors.

Kinematic, same npz format as reference 1 (`cartwheel_ref.py`): the base height
puts the lowest point (limb tips, or the chassis if it would go lower) on the
floor, and travel follows the pivot tip without slipping.

  python cartwheel_ref2.py   # writes tasks/jumper/cartwheel_flat/ref/cartwheel_flat_crab.npz
"""
from __future__ import annotations

import json
import math
import pathlib

import mujoco
import numpy as np

from tools.cartwheel.refs.cartwheel_ref import CHASSIS, FEET, LEGS, TIPS, mj
from tools.crab_trio.stage import HOME, JOINTS

ROOT = pathlib.Path(__file__).resolve().parents[3]
XML = ROOT / "assets/jumper/jumper.xml"
OUT = ROOT / "tasks/jumper/cartwheel_flat/ref/cartwheel_flat_crab.npz"

REC_HZ, CONTROL_HZ, T_GO = 250.0, 200.0, 0.3

#: the flat six-spoke star: hip yaw (rad) per limb, everything else straight
#: (found with a scan of each hip-yaw range: spokes at 30/90/150/210/270/330 deg,
#: radius 0.29-0.32 m, tips within 3.5 cm of the body plane)
FLAT_YAW = {"LF": -1.160, "LM": -0.030, "LR": 1.118, "RR": -1.113, "RM": 0.036, "RF": 1.160}

# timeline after go (s)
T_TIP, T_ROLL, T_DOWN, T_END = 0.6, 2.1, 2.7, 3.0


def flat_star() -> np.ndarray:
    p = {j: 0.0 for j in JOINTS}
    for L, y in FLAT_YAW.items():
        p[f"{L}_J0_joint"] = y
    return np.array([p[j] for j in JOINTS])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def schedule(tsg, home, flat):
    """(joints, tip-onto-side angle phi about world x, wheel angle theta about body z)."""
    if tsg < 0:
        return home, 0.0, 0.0
    if tsg < T_TIP:
        a = tsg / T_TIP
        return home + mj((a - 0.25) / 0.75) * (flat - home), math.radians(90) * mj(a), 0.0
    if tsg < T_ROLL:
        s = (tsg - T_TIP) / (T_ROLL - T_TIP)
        return flat, math.radians(90), 2 * math.pi * (s - math.sin(2 * math.pi * s) / (2 * math.pi))
    if tsg < T_DOWN:
        a = (tsg - T_ROLL) / (T_DOWN - T_ROLL)
        return flat + mj(a / 0.75) * (home - flat), math.radians(90) * (1 - mj(a)), 2 * math.pi
    return home, 0.0, 2 * math.pi


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

    def place(q, quat, x, y, z=0.0):
        d.qpos[:] = 0
        d.qpos[0:3] = [x, y, z]
        d.qpos[3:7] = quat
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)

    def chassis_min_z():
        return min(float(((d.geom_xmat[g].reshape(3, 3) @ v.T)[2] + d.geom_xpos[g][2]).min())
                   for g, v in chassis)

    place(home, [1, 0, 0, 0], 0, 0)
    r_foot = 0.10647 + min(d.xpos[b][2] for b in tips)

    n = int(round((T_GO + T_END) * REC_HZ))
    t = np.arange(n) / REC_HZ
    Q, BP, BQ = np.zeros((n, len(JOINTS))), np.zeros((n, 3)), np.zeros((n, 4))
    FP, AV = np.zeros((n, 6, 3)), np.zeros((n, 3))
    x = y = 0.0
    pivot = pivot_xy = None
    chassis_touch = 0
    prev_quat = None
    for i, ti in enumerate(t):
        q, phi, theta = schedule(ti - T_GO, home, flat)
        q = np.clip(q, lo, hi)
        quat = quat_mul([math.cos(phi / 2), math.sin(phi / 2), 0, 0],
                        [math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
        place(q, quat, x, y)
        tz = np.array([d.xpos[b][2] for b in tips])
        k = int(np.argmin(tz))
        if k == pivot:                                 # no slip at the pivot tip
            x += pivot_xy[0] - d.xpos[tips[k]][0]
            y += pivot_xy[1] - d.xpos[tips[k]][1]
            place(q, quat, x, y)
        pivot, pivot_xy = k, d.xpos[tips[k]][:2].copy()
        z_tips = r_foot - tz.min()
        cz = chassis_min_z()                           # with z = 0
        z = max(z_tips, -cz)                           # never through the floor
        if -cz > z_tips + 1e-4 and T_GO + 0.05 < ti < T_GO + T_END - 0.05:
            chassis_touch += 1
        place(q, quat, x, y, z)
        Q[i], BP[i], BQ[i] = q, d.xpos[base], d.xquat[base]
        FP[i] = [d.xpos[b] for b in feet]
        if prev_quat is not None:                      # world angular velocity from the quat step
            dq = quat_mul(quat, [prev_quat[0], -prev_quat[1], -prev_quat[2], -prev_quat[3]])
            ang = 2 * math.atan2(np.linalg.norm(dq[1:]), dq[0])
            ax = dq[1:] / (np.linalg.norm(dq[1:]) + 1e-12)
            AV[i] = ax * ang * REC_HZ
        prev_quat = quat
    dt = 1.0 / REC_HZ
    QD = np.gradient(Q, dt, axis=0)
    LV = np.gradient(BP, dt, axis=0)
    contact = (FP[:, :, 2] - FP[:, :, 2].min(0, keepdims=True) < 0.010)
    qc_t = np.arange(int(round(t[-1] * CONTROL_HZ)) + 1) / CONTROL_HZ
    QC = np.stack([np.interp(qc_t, t, Q[:, j]) for j in range(len(JOINTS))], 1)
    roll_mask = (t > T_GO + T_TIP) & (t < T_GO + T_ROLL)
    meta = dict(
        description="kinematic flat-crab cartwheel: six straight limbs spread in the body plane, "
                    "rolling on edge (tools/cartwheel/refs/cartwheel_ref2.py)",
        joint_order=list(JOINTS), leg_order=list(LEGS), rec_hz=REC_HZ, control_hz=CONTROL_HZ,
        t_go=T_GO, go_frame=int(round(T_GO * REC_HZ)), stand_base_z=float(BP[0, 2]),
        kp=20.0, kd=0.5, tau_limit=1.7464, kinematic=True,
        stats=dict(peak_wheel_rate=float(np.linalg.norm(AV, axis=1).max()),
                   travel_x=float(BP[-1, 0] - BP[0, 0]), travel_y=float(BP[-1, 1] - BP[0, 1]),
                   peak_base_z=float(BP[:, 2].max()),
                   base_z_range_while_rolling=[float(BP[roll_mask, 2].min()), float(BP[roll_mask, 2].max())],
                   chassis_on_floor_s=chassis_touch / REC_HZ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, meta=json.dumps(meta), t=t, q=Q, qd=QD, q_cmd=QC, base_pos=BP, base_quat=BQ,
             base_lin_vel=LV, base_ang_vel=AV, foot_pos=FP, contact=contact)
    print("wrote", OUT)
    print(json.dumps(meta["stats"], indent=1))


if __name__ == "__main__":
    build()
