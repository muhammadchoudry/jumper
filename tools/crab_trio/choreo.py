"""'Reef Rave' -- an original 26-second routine for three Neon Reef Jumpers.

120 BPM (one beat = 0.5 s). Each crab gets a list of keyframes; the servo
targets glide between them with a minimum-jerk curve and are tracked by the
robot's own PD servos in MuJoCo physics. No learned policy, no kinematic
playback: if a pose is not physically balanced, the crab falls over.

Sections
  A  intro        all three settle, then widen into the dance stance
  B  ripple       claws go up crab by crab, left to right, then come down the same way
  C  clicks       all three point claws forward and snap their pincers on the beat
  D  pendulum     claws wide, bodies twist; the middle crab mirrors its neighbours
  E  pump         alternating claw pumps with a bounce; middle crab in counter-phase
  F  reverse      claws raised high crab by crab, right to left
  J  jumps        real jumps: one crab at a time left to right, then all three together
  G  finale       unison bounces with claws high, then a bow back to home
"""
from __future__ import annotations

import jump
from stage import HOME

BEAT = 0.5
N_CRABS = 3

# ---- poses (absolute joint targets) --------------------------------------
# Leg stance from the robot's own claws-up dance frames: middle and rear legs
# swung forward under the body so it stays balanced without its front feet.
STANCE = {
    "LM_J0_joint": -0.52, "LM_J1_joint": 0.38, "LM_J2_joint": -1.69,
    "RM_J0_joint": 0.53, "RM_J1_joint": -0.38, "RM_J2_joint": 1.69,
    "LR_J0_joint": 0.06, "LR_J1_joint": 0.41, "LR_J2_joint": -1.77,
    "RR_J0_joint": -0.06, "RR_J1_joint": -0.41, "RR_J2_joint": 1.77,
}
HOME_LEGS = {j: v for j, v in HOME.items() if not j.startswith(("LF", "RF"))}

# Left-claw poses: (J0, J1, J2, J3, J4). The right claw is the mirror image.
ARM = {
    "down":  (-0.55, -1.07, -0.44, -1.37, 0.0),   # resting on the floor (home)
    "up":    (-0.52, -2.79, -1.57, -0.17, 0.0),   # raised, folded over the shell
    "high":  (-0.40, -3.10, -1.57, 0.00, 0.0),    # straight up
    "front": (-0.30, -1.90, -0.60, -0.20, 0.0),   # pointing forward
    "wide":  (0.30, -2.20, -0.20, -0.30, 0.0),    # spread out to the side
}
SNAP = -1.4  # J4 offset: pincer open


def arm(side: str, name: str, snap: bool = False) -> dict[str, float]:
    j0, j1, j2, j3, j4 = ARM[name]
    if snap:
        j4 += SNAP
    if side == "L":
        vals = (j0, j1, j2, j3, j4)
    else:  # mirror: everything but the shoulder lift flips sign
        vals = (-j0, j1, -j2, -j3, -j4)
    return {f"{side}F_J{k}_joint": v for k, v in enumerate(vals)}


def body(legs: dict[str, float], bounce: float = 0.0, twist: float = 0.0) -> dict[str, float]:
    """bounce > 0 crouches the body; twist rotates it about the vertical axis."""
    out = dict(legs)
    for leg in ("LM", "LR", "RM", "RR"):
        s = 1 if leg[0] == "L" else -1
        out[f"{leg}_J1_joint"] += s * bounce
        out[f"{leg}_J2_joint"] += s * bounce
        out[f"{leg}_J0_joint"] += twist
    return out


def pose(left="down", right="down", legs=STANCE, bounce=0.0, twist=0.0,
         snap_l=False, snap_r=False) -> dict[str, float]:
    return {**body(legs, bounce, twist), **arm("L", left, snap_l), **arm("R", right, snap_r)}


# ---- the routine ----------------------------------------------------------
def routine(k: int) -> list[tuple[float, float, dict[str, float]]]:
    """Keyframes for crab k (0 = left of frame): (start_beat, glide_beats, pose)."""
    mid = k == 1
    sign = -1 if mid else 1
    kf: list[tuple[float, float, dict[str, float]]] = []
    add = lambda beat, glide, p: kf.append((beat, glide, p))  # noqa: E731

    # A intro: settle, then widen the legs into the stance (claws stay down)
    add(0, 0.01, pose(legs=HOME_LEGS))
    add(3, 2.0, pose())

    # B ripple up then down, left to right
    add(6 + k, 0.8, pose("up", "up"))
    add(10 + k, 0.8, pose())

    # C clicks: claws forward, snap on every beat for 6 beats
    add(14, 1.0, pose("front", "front"))
    for i in range(6):
        b = 15 + i
        add(b, 0.25, pose("front", "front", snap_l=True, snap_r=True))
        add(b + 0.5, 0.25, pose("front", "front"))

    # D pendulum: claws wide, twist left/right every 2 beats (middle mirrored)
    add(21, 1.0, pose("wide", "wide"))
    for i in range(4):
        add(22 + 2 * i, 1.6, pose("wide", "wide", twist=sign * (0.22 if i % 2 == 0 else -0.22)))
    add(30, 0.8, pose())

    # E pump: alternate claws each beat with a bounce; middle crab counter-phase
    for i in range(8):
        left_up = (i % 2 == 0) != mid
        add(31 + i, 0.4, pose("up" if left_up else "down", "down" if left_up else "up",
                              bounce=0.06))
        add(31.5 + i, 0.4, pose("up" if left_up else "down", "down" if left_up else "up"))

    # F reverse ripple: claws high, right to left
    add(39, 0.8, pose())
    add(40 + (2 - k), 0.8, pose("high", "high"))

    # J jumps: settle into the jump stance; the jumps themselves are driven by
    #   jump.py (see JUMPS below) -- here each crab just holds that stance.
    add(44, 1.5, dict(jump.STAND))

    # G finale: unison bounces, then the bow
    add(57.5, 1.0, pose())                  # legs back under first, claws still down
    add(58.75, 1.0, pose("high", "high"))
    for i in range(3):
        add(60 + i, 0.4, pose("high", "high", bounce=0.14))
        add(60.5 + i, 0.4, pose("high", "high"))
    add(63.5, 1.0, pose("down", "down", bounce=0.12))   # bow
    add(65.5, 1.5, pose(legs=HOME_LEGS))                # back to home
    return kf


END_BEAT = 69

# Jump apexes (beats): a left-to-right ripple, then all three together.
APEX_DELAY = 0.864 / BEAT        # takeoff command -> apex, from the jump data
RIPPLE_APEX = [48, 50, 52]
UNISON_APEX = 56


def jumps(k: int) -> list[float]:
    """Times (s) at which crab k starts its jump command sequence."""
    return [(b - APEX_DELAY) * BEAT for b in (RIPPLE_APEX[k], UNISON_APEX)]


def _minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def targets(kf, t: float) -> dict[str, float]:
    """Servo targets at time t (s) for one crab's keyframe list."""
    cur = dict(HOME)
    for beat, glide, p in kf:
        start = beat * BEAT
        if t < start:
            break
        a = _minjerk((t - start) / max(glide * BEAT, 1e-6))
        cur = {j: cur[j] + a * (p.get(j, cur[j]) - cur[j]) for j in cur}
    return cur
