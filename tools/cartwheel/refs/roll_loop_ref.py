"""Spawn states for `jumper.cartwheel_roll`: the flat star on its edge, rolling.

Not a motion to copy -- that task's policy is free to move its legs however rolls
best. This is what episodes start from (reference-state initialisation) and what
the phase observation counts along: the six-spoke star on its right edge, rolling
forward at a steady ROLL_RATE for TURNS full turns. Kinematic, same npz format as
references 1-4 (lowest point on the floor, no slip at the pivot tip).

  python roll_loop_ref.py    # writes tasks/jumper/cartwheel_roll/ref/roll_loop.npz
"""
from __future__ import annotations

import json
import math
import pathlib

import mujoco
import numpy as np

from tools.cartwheel.refs.cartwheel_ref import FEET, LEGS, TIPS
from tools.cartwheel.refs.cartwheel_ref2 import flat_star, quat_mul
from tools.crab_trio.stage import JOINTS

ROOT = pathlib.Path(__file__).resolve().parents[3]
XML = ROOT / "assets/jumper/jumper.xml"
OUT = ROOT / "tasks/jumper/cartwheel_roll/ref/roll_loop.npz"

REC_HZ, CONTROL_HZ, T_GO = 250.0, 200.0, 0.02
ROLL_RATE = 2.7          # rad/s (~0.8 m/s at the rim): the speed the run-up and jump give
TURNS = 3.0
THETA0 = math.radians(-15)


def build():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
    lo = np.array([m.jnt_range[m.joint(j).id][0] for j in JOINTS])
    hi = np.array([m.jnt_range[m.joint(j).id][1] for j in JOINTS])
    base = m.body("base_link").id
    tips = [m.body(b).id for b in TIPS]
    feet = [m.body(b).id for b in FEET]
    star = np.clip(flat_star(), lo, hi)

    def place(quat, x, y, z=0.0):
        d.qpos[:] = 0
        d.qpos[0:3] = [x, y, z]
        d.qpos[3:7] = quat
        d.qpos[qa] = star
        mujoco.mj_kinematics(m, d)

    place([1, 0, 0, 0], 0, 0)
    r_foot = 0.0105
    t_end = T_GO + TURNS * 2 * math.pi / ROLL_RATE
    n = int(round(t_end * REC_HZ))
    t = np.arange(n) / REC_HZ
    Q = np.tile(star, (n, 1))
    BP, BQ, FP = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, 6, 3))
    x = y = 0.0
    pivot = pivot_xy = None
    for i, ti in enumerate(t):
        theta = THETA0 + ROLL_RATE * max(ti - T_GO, 0.0)
        quat = quat_mul([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
                        [math.cos(theta / 2), 0, 0, -math.sin(theta / 2)])      # rolls forward (+x)
        place(quat, x, y)
        tz = np.array([d.xpos[b][2] for b in tips])
        k = int(np.argmin(tz))
        if k == pivot:
            x += pivot_xy[0] - d.xpos[tips[k]][0]
            y += pivot_xy[1] - d.xpos[tips[k]][1]
            place(quat, x, y)
        pivot, pivot_xy = k, d.xpos[tips[k]][:2].copy()
        place(quat, x, y, r_foot - tz.min())
        BP[i], BQ[i] = d.xpos[base], d.xquat[base]
        FP[i] = [d.xpos[b] for b in feet]
    dt = 1 / REC_HZ
    AV = np.zeros((n, 3))
    for i in range(1, n):
        dq = quat_mul(BQ[i], [BQ[i - 1][0], -BQ[i - 1][1], -BQ[i - 1][2], -BQ[i - 1][3]])
        if dq[0] < 0:
            dq = -dq
        ang = 2 * math.atan2(np.linalg.norm(dq[1:]), dq[0])
        AV[i] = dq[1:] / (np.linalg.norm(dq[1:]) + 1e-12) * ang * REC_HZ
    AV[0] = AV[1]
    LV = np.gradient(BP, dt, axis=0)
    k = 25                                     # smooth the pivot-change kinks in the hub velocity
    LV = np.stack([np.convolve(np.pad(LV[:, j], (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid")
                   for j in range(3)], 1)[:n]
    QD = np.zeros_like(Q)
    contact = (FP[:, :, 2] - FP[:, :, 2].min(0, keepdims=True) < 0.010)
    qc_t = np.arange(int(round(t[-1] * CONTROL_HZ)) + 1) / CONTROL_HZ
    QC = np.tile(star, (len(qc_t), 1))
    meta = dict(
        description="spawn states for jumper.cartwheel_roll: flat six-spoke star on its edge rolling "
                    f"forward at {ROLL_RATE} rad/s for {TURNS} turns (tools/cartwheel/refs/roll_loop_ref.py)",
        joint_order=list(JOINTS), leg_order=list(LEGS), rec_hz=REC_HZ, control_hz=CONTROL_HZ,
        t_go=T_GO, go_frame=int(round(T_GO * REC_HZ)), stand_base_z=float(BP[0, 2]),
        kp=20.0, kd=0.5, tau_limit=1.7464, kinematic=True,
        stats=dict(roll_rate=ROLL_RATE, travel=float(BP[-1, 0] - BP[0, 0]),
                   hub_z=[float(BP[:, 2].min()), float(BP[:, 2].max())], duration=float(t[-1])),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, meta=json.dumps(meta), t=t, q=Q, qd=QD, q_cmd=QC, base_pos=BP, base_quat=BQ,
             base_lin_vel=LV, base_ang_vel=AV, foot_pos=FP, contact=contact)
    print("wrote", OUT)
    print(json.dumps(meta["stats"], indent=1))


if __name__ == "__main__":
    build()
