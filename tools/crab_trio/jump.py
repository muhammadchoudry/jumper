"""Open-loop replay of the repo's high-jump servo commands, under physics.

tasks/jumper/jump/ref/high_jump_flat.npz stores the joint commands (q_cmd, 200 Hz)
that produced a 14 cm jump with kp 20 / kd 0.5 and a torque-speed motor
envelope. Here those commands drive a crab's joints through the same motor
model, computed as joint torques -- nothing about the body is scripted.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_D = np.load(ROOT / "tasks/jumper/jump/ref/high_jump_flat.npz", allow_pickle=True)
META = json.loads(str(_D["meta"]))
Q_CMD = _D["q_cmd"]                    # (n, 22) at control_hz
ORDER = META["joint_order"]
CTRL_DT = 1.0 / META["control_hz"]
KP, KD = META["kp"], META["kd"]
STAND = META["stand_pose"]
DURATION = len(Q_CMD) * CTRL_DT        # ~1.47 s

_M = META["motor"]
_RPM = 60 / (2 * np.pi)


def torque_limit(qd: np.ndarray) -> np.ndarray:
    """Torque-speed envelope: flat plateau, then falling to zero at the cap speed."""
    rpm = np.abs(qd) * _RPM
    frac = np.clip((_M["cap_rpm"] - rpm) / (_M["cap_rpm"] - _M["corner_rpm"]), 0, 1)
    return np.where(rpm <= _M["corner_rpm"], _M["P"], _M["P"] * frac)


def command(t: float) -> dict[str, float]:
    """Jump servo targets t seconds after the jump starts (holds the last frame after)."""
    i = min(int(t / CTRL_DT), len(Q_CMD) - 1)
    return dict(zip(ORDER, Q_CMD[i]))


def torque(target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    tau = KP * (target - q) - KD * qd
    lim = torque_limit(qd)
    return np.clip(tau, -lim, lim)
