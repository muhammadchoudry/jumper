"""Straight-limb poses: let the bones carry the load, not the motors.

At zero joint angles every Jumper leg points straight out sideways, and the hip
(J1) and knee (J2) of the walking legs -- and the J2/J3 of the claws -- all swing
in the roll plane. So a limb is a straight spoke when its knee is zero, and the
hip angle alone aims it. A load along a straight limb passes (nearly) through its
joints, so it needs far less torque than a bent limb holding the same weight.

spoke(L, deg): joint targets that aim limb L straight along `deg` in the roll
plane, measured from the body's left (+y) toward up (+z), like the hip itself.
"""
from __future__ import annotations

import math

from stage import HOME, JOINTS

LEGS3 = ("LM", "LR", "RM", "RR")
CLAWS = ("LF", "RF")


def spoke(limb: str, deg: float, yaw: float = 0.0) -> dict[str, float]:
    """Joint targets for limb pointing straight at `deg` (left side: 0 = out, 90 = up).

    For a right-side limb `deg` is measured the same way from the body's +y axis,
    so 180 is straight out to the right and 90 is up. `yaw` (deg, + = forward)
    fans the limb forward or back with the hip-yaw joint; the limb stays straight.
    """
    y = math.radians(yaw) * (-1.0 if limb[0] == "L" else 1.0)
    a = math.radians(deg)
    out: dict[str, float] = {}
    if limb[0] == "L":
        q = a                                  # +rotation about +x lifts a left limb
    else:
        q = -(math.pi - a)                     # mirror image of a left limb at (180 - deg)
        q = (q + math.pi) % (2 * math.pi) - math.pi
    if limb in LEGS3:
        out[f"{limb}_J0_joint"] = y
        out[f"{limb}_J1_joint"] = q
        out[f"{limb}_J2_joint"] = 0.0
    else:
        out[f"{limb}_J0_joint"] = y
        out[f"{limb}_J1_joint"] = 0.0
        out[f"{limb}_J2_joint"] = q
        out[f"{limb}_J3_joint"] = 0.0
        out[f"{limb}_J4_joint"] = 0.0
    return out


def pose(spokes: dict, base: dict[str, float] | None = None) -> dict[str, float]:
    """spokes: {limb: deg} or {limb: (deg, yaw)}."""
    p = dict(base or HOME)
    for limb, v in spokes.items():
        deg, yaw = v if isinstance(v, tuple) else (v, 0.0)
        p.update(spoke(limb, deg, yaw))
    return p


__all__ = ["spoke", "pose", "JOINTS"]
