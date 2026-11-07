"""Author the reference motion for the `jumper.cartwheel` training task.

A kinematic cartwheel on straight spokes, in the npz format the repo's jump
reference uses (tasks/jumper/jump/mdp/reference.py):

  stand -> fanned crouch -> push -> limbs snap into the straight-spoke star ->
  roll 360 degrees sideways over the spokes -> back to the stance -> home

The body pose is solved from the limbs, not invented: at every frame the base
height puts the lowest limb tip on the floor, and the sideways travel follows
the pivot tip without slipping. It is *kinematic* -- gravity and torque are not
enforced here; that is what the RL policy is for (it learns residual corrections
that make the motion work under physics, starting from random points along it).

  python cartwheel_ref.py              # writes tasks/jumper/cartwheel/ref/cartwheel_flat.npz
"""
from __future__ import annotations

import json
import math
import pathlib

import mujoco
import numpy as np

from tools.cartwheel.physics import straight as S
from tools.cartwheel.physics import wheel_s as W
from tools.cartwheel.physics import wheel_s2 as W2
from tools.crab_trio.stage import HOME, JOINTS

ROOT = pathlib.Path(__file__).resolve().parents[3]
XML = ROOT / "assets/jumper/jumper.xml"
OUT = ROOT / "tasks/jumper/cartwheel/ref/cartwheel_flat.npz"

REC_HZ = 250.0
CONTROL_HZ = 200.0
T_GO = 0.3
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")
FEET = ("LF_palm_pad_b_link", "RF_palm_pad_b_link", "LM_foot_tip_link",
        "RM_foot_tip_link", "LR_foot_tip_link", "RR_foot_tip_link")
TIPS = ("LF_finger_tip_link", "RF_finger_tip_link", "LM_foot_tip_link",
        "RM_foot_tip_link", "LR_foot_tip_link", "RR_foot_tip_link")
CHASSIS = ("base_link", "upper_shell_link", "camera_link", "tof_sensor_link", "display_module_link")


def mj(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10 - 15 * t + 6 * t * t)


def poses():
    W._setup()
    star = dict(zip(JOINTS, W.target(W.STAR)))      # straight-spoke star (wheel_s.py)
    stance = dict(zip(JOINTS, W.target(W.STANCE)))
    stance_v = np.array([stance[j] for j in JOINTS])
    crouch_v = W2._offset(stance_v, np.array([[0, 0]] * 2 + [list(W2.CROUCH)] * 4, dtype=float))
    # push: the left legs drive straight down (lifting the left side: +roll), right legs fold
    push = S.pose({"LM": (-75, 30), "LR": (-75, -30), "RM": (230, 30), "RR": (230, -30),
                   "LF": (60, 25), "RF": (120, 25)})
    push_v = np.clip([push[j] for j in JOINTS], W.LLO, W.LHI)
    return (np.array([HOME[j] for j in JOINTS]), crouch_v, push_v,
            np.array([star[j] for j in JOINTS]), stance_v)


def schedule(tsg, P):
    """(joint angles, roll angle) at time since go."""
    home, crouch, push, star, stance = P
    if tsg < 0:
        return home, 0.0
    if tsg < 0.45:
        return home + mj(tsg / 0.45) * (crouch - home), 0.0
    if tsg < 0.65:
        a = (tsg - 0.45) / 0.20
        return crouch + mj(a) * (push - crouch), math.radians(25) * a * a
    if tsg < 1.75:
        a = (tsg - 0.65) / 1.10
        q = push + mj((tsg - 0.65) / 0.12) * (star - push)
        # 25 -> 360 deg; rate continuous with the push (~4.4 rad/s), easing into the landing
        phi = math.radians(25) + math.radians(335) * (a + 0.12 * math.sin(math.pi * a) / math.pi * -1)
        return q, phi
    if tsg < 2.15:
        a = (tsg - 1.75) / 0.40
        q = star + mj(2 * a) * (stance - star) if a < 0.5 else stance + mj(2 * a - 1) * (home - stance)
        return q, 2 * math.pi
    return home, 2 * math.pi


def build():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
    base = m.body("base_link").id
    tips = [m.body(b).id for b in TIPS]
    feet = [m.body(b).id for b in FEET]
    chassis_geoms = [g for g in range(m.ngeom)
                     if m.body(m.geom_bodyid[g]).name in CHASSIS and m.geom_contype[g]]
    P = poses()

    def place(q, phi, y):
        d.qpos[:] = 0
        d.qpos[1] = y
        d.qpos[3:7] = [math.cos(phi / 2), math.sin(phi / 2), 0, 0]
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)

    # foot radius: the tip-origin height when standing at HOME on the floor
    place(P[0], 0.0, 0.0)
    r_foot = 0.10647 + min(d.xpos[b][2] for b in tips)

    n = int(round((T_GO + 2.4) * REC_HZ))
    t = np.arange(n) / REC_HZ
    Q = np.zeros((n, len(JOINTS)))
    BP = np.zeros((n, 3))
    BQ = np.zeros((n, 4))
    FP = np.zeros((n, 6, 3))
    PHI = np.zeros(n)
    y = 0.0
    pivot, pivot_y = None, None
    min_chassis = 1.0
    for i, ti in enumerate(t):
        q, phi = schedule(ti - T_GO, P)
        q = np.clip(q, W.LLO, W.LHI)
        place(q, phi, y)
        tz = np.array([d.xpos[b][2] for b in tips])
        k = int(np.argmin(tz))
        if k == pivot:                      # roll over the pivot tip without slipping
            y += pivot_y - d.xpos[tips[k]][1]
            place(q, phi, y)
        pivot, pivot_y = k, d.xpos[tips[k]][1]
        z = r_foot - tz.min()
        d.qpos[2] = z
        mujoco.mj_kinematics(m, d)
        Q[i], PHI[i] = q, phi
        BP[i] = d.xpos[base]
        BQ[i] = d.xquat[base]
        FP[i] = [d.xpos[b] for b in feet]
        # chassis clearance: lowest point of the chassis collision meshes
        for g in chassis_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                wz = (d.geom_xmat[g].reshape(3, 3) @ v.T)[2] + d.geom_xpos[g][2]
                if T_GO + 0.6 < ti < T_GO + 1.75:
                    min_chassis = min(min_chassis, float(wz.min()))
    dt = 1.0 / REC_HZ
    QD = np.gradient(Q, dt, axis=0)
    LV = np.gradient(BP, dt, axis=0)
    roll_rate = np.gradient(PHI, dt)
    AV = np.stack([roll_rate, np.zeros(n), np.zeros(n)], 1)    # world frame, about x
    contact = (FP[:, :, 2] - FP[:, :, 2].min(0, keepdims=True) < 0.010)
    q_cmd_t = np.arange(int(round(t[-1] * CONTROL_HZ)) + 1) / CONTROL_HZ
    QC = np.stack([np.interp(q_cmd_t, t, Q[:, j]) for j in range(len(JOINTS))], 1)
    go_frame = int(round(T_GO * REC_HZ))
    meta = dict(
        description="kinematic straight-spoke cartwheel (tools/cartwheel/refs/cartwheel_ref.py)",
        joint_order=list(JOINTS), leg_order=list(LEGS), rec_hz=REC_HZ, control_hz=CONTROL_HZ,
        t_go=T_GO, go_frame=go_frame, stand_base_z=float(BP[0, 2]), kp=20.0, kd=0.5,
        tau_limit=1.7464, kinematic=True,
        stats=dict(peak_roll_rate=float(np.abs(roll_rate).max()), sideways_travel=float(BP[-1, 1] - BP[0, 1]),
                   min_chassis_height_mid_roll=min_chassis, peak_base_z=float(BP[:, 2].max())),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, meta=json.dumps(meta), t=t, q=Q, qd=QD, q_cmd=QC, base_pos=BP, base_quat=BQ,
             base_lin_vel=LV, base_ang_vel=AV, foot_pos=FP, contact=contact, roll=PHI)
    print("wrote", OUT)
    print(json.dumps(meta["stats"], indent=1))
    return OUT


if __name__ == "__main__":
    build()
