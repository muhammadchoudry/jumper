"""Build a MuJoCo scene with several Neon Reef Jumpers on a dark neon stage.

Each robot gets a position actuator per joint with the robot's own servo
numbers (kp 10, kd 0.5, torque limit 1.75 N*m, from tasks/jumper/common/constants.py),
so the choreography is physically simulated, not keyframe-animated.
"""
from __future__ import annotations

import math
import pathlib

import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKIN = ROOT / "assets/jumper/jumper_neon_reef.xml"

KP, KD, TORQUE = 10.0, 0.5, 1.7464

HOME = {
    "LF_J0_joint": -0.5507, "LF_J1_joint": -1.0658,
    "LF_J2_joint": -0.4408, "LF_J3_joint": -1.374, "LF_J4_joint": 0.0,
    "RF_J0_joint": 0.5524, "RF_J1_joint": -1.066,
    "RF_J2_joint": 0.4414, "RF_J3_joint": 1.4173, "RF_J4_joint": 0.0,
    "LM_J0_joint": 0.0012, "LM_J1_joint": 0.5356, "LM_J2_joint": -1.8271,
    "RM_J0_joint": -0.0012, "RM_J1_joint": -0.5376, "RM_J2_joint": 1.8313,
    "LR_J0_joint": 0.6313, "LR_J1_joint": 0.5534, "LR_J2_joint": -1.8648,
    "RR_J0_joint": -0.6326, "RR_J1_joint": -0.5548, "RR_J2_joint": 1.8678,
}
JOINTS = list(HOME)
STAND_Z = 0.10647


def build(placements: list[tuple[float, float, float]], torque_control: bool = False) -> mujoco.MjModel:
    """placements: (x, y, yaw_deg) per robot. Robot i's names get prefix c{i}_."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.07, 0.07, 0.10]
    spec.visual.headlight.diffuse = [0.14, 0.14, 0.18]

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.08, 0.03, 0.14], rgb2=[0, 0, 0.02], width=256, height=256)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.035, 0.04, 0.08], rgb2=[0.06, 0.07, 0.13], width=512, height=512)
    mat = spec.add_material(name="floor", texrepeat=[24, 24], texuniform=True, reflectance=0.25)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"

    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   material="floor", friction=[1.0, 0.005, 0.0001])
    for name, pos, d, rgb, shadow in [
        ("key", [-2, -2, 3], [0.5, 0.5, -1], [0.5, 0.52, 0.6], True),
        ("rim_cyan", [3, 2, 1.5], [-0.7, -0.4, -0.6], [0.0, 0.35, 0.45], False),
        ("rim_magenta", [-3, 2, 1.5], [0.7, -0.4, -0.6], [0.45, 0.05, 0.35], False),
    ]:
        light = world.add_light(name=name, pos=pos, dir=d, diffuse=rgb, castshadow=shadow)
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    for i, (x, y, yaw) in enumerate(placements):
        robot = mujoco.MjSpec.from_file(str(SKIN))
        frame = world.add_frame(pos=[x, y, STAND_Z + 0.004],
                                quat=[math.cos(math.radians(yaw) / 2), 0, 0,
                                      math.sin(math.radians(yaw) / 2)])
        frame.attach_body(robot.body("base_link"), f"c{i}_", "")
        for j in JOINTS:
            act = spec.add_actuator(name=f"c{i}_{j}", target=f"c{i}_{j}",
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT)
            if torque_control:  # the caller computes PD torque itself
                act.set_to_motor()
                act.ctrllimited = True
                act.ctrlrange = [-TORQUE, TORQUE]
            else:
                act.set_to_position(kp=KP, kv=KD)
                act.forcelimited = True
                act.forcerange = [-TORQUE, TORQUE]
    # attached sub-models carry their own textures/materials; compile once
    return spec.compile()


def qpos_addr(model: mujoco.MjModel, robot: int) -> dict[str, int]:
    return {j: model.jnt_qposadr[model.joint(f"c{robot}_{j}").id] for j in JOINTS}


def reset_home(model: mujoco.MjModel, data: mujoco.MjData, n: int) -> None:
    for i in range(n):
        for j, a in qpos_addr(model, i).items():
            data.qpos[a] = HOME[j]
        for j in JOINTS:
            data.ctrl[model.actuator(f"c{i}_{j}").id] = HOME[j]
    mujoco.mj_forward(model, data)
