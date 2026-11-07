"""Render a recorded rollout (roll_record.py's npz) to an mp4 + 8-frame contact sheet,
on a checker floor.

  MUJOCO_GL=egl python -m tools.cartwheel.render.traj_video roll_traj.npz out.mp4
  TOPVIEW=1 ...                                   # a high fixed view over the whole path
  COURSE=1|long ...                               # a ramp-jump course (tasks/jumper/cartwheel_stunt)
  SLOPE_DEG=8 ...                                 # a run simulated with gravity tilted toward +x
                                                  # (jumper.cartwheel_hill): drawn on a real hill
"""
import sys
import numpy as np
import mujoco
import imageio
D = np.load(sys.argv[1]); out = sys.argv[2]
spec = mujoco.MjSpec.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[3] / "assets/jumper/jumper.xml"))
import math as _m
SLOPE = _m.radians(float(__import__("os").environ.get("SLOPE_DEG", "0")))
# world = R_y(SLOPE) * sim frame: tilted gravity becomes straight down, the floor a hill
# going down toward +x
QS = np.array([_m.cos(SLOPE / 2), 0.0, _m.sin(SLOPE / 2), 0.0])
RS = np.array([[_m.cos(SLOPE), 0, _m.sin(SLOPE)], [0, 1, 0], [-_m.sin(SLOPE), 0, _m.cos(SLOPE)]])
tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       rgb1=[0.12, 0.14, 0.2], rgb2=[0.2, 0.22, 0.3], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[8, 8]); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], material="grid", contype=0, conaffinity=0,
                        quat=QS)
_cname = __import__("os").environ.get("COURSE", "")
COURSE = _cname in ("1", "default", "long")
if COURSE:
    from tasks.jumper.cartwheel_stunt import course as _cm
    _course = _cm.COURSES["long" if _cname == "long" else "default"]
    # drawn 1.2 m wide instead of the simulated 6 m, and slid sideways to stay under the
    # robot frame by frame (it is the same all the way across, and the run stays inside the
    # real width), so a side camera sees the hill, kicker and gap
    spec.add_material(name="ramp", rgba=[0.62, 0.52, 0.40, 1.0])     # plywood, against the dark pit floor
    _course.add_course(spec, material="ramp", contype=0, conaffinity=0, y0=0.0, half_width=0.6)
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
if SLOPE or COURSE:   # a long hill: a sun as well as the spot at the top
    spec.worldbody.add_light(pos=[0, 0, 10], dir=[0.3, 0.4, -1], diffuse=[0.55, 0.55, 0.6], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
m = spec.compile()
COURSE_GEOMS = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("course_")] if COURSE else []
d = mujoco.MjData(m)
qa = [m.jnt_qposadr[m.joint(str(n)).id] for n in D["names"]]
r = mujoco.Renderer(m, 480, 854); cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
wr = imageio.get_writer(out, fps=30, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
dt = float(D["dt"]); step = max(1, int(round(1 / 30 / dt))); frames = []
import os
TOP = os.environ.get("TOPVIEW") == "1"
cx, cy = (D["qs"][:, 0].min() + D["qs"][:, 0].max()) / 2, (D["qs"][:, 1].min() + D["qs"][:, 1].max()) / 2
TOP_DIST = 1.6 * max(np.ptp(D["qs"][:, 0]), np.ptp(D["qs"][:, 1]), 1.5) + 0.8
idx = list(range(0, len(D["qs"]), step)); picks = set(idx[int(i * (len(idx) - 1) / 7)] for i in range(8))
for k in range(0, len(D["qs"]), step):
    q = D["qs"][k].copy()
    if SLOPE:
        q[0:3] = RS @ q[0:3]
        qq = np.zeros(4); mujoco.mju_mulQuat(qq, QS, q[3:7]); q[3:7] = qq
    d.qpos[:] = 0; d.qpos[0:3] = q[0:3]; d.qpos[3:7] = q[3:7]; d.qpos[qa] = q[7:]
    for _g in COURSE_GEOMS:
        m.geom_pos[_g, 1] = q[1]
    mujoco.mj_forward(m, d)
    if TOP:                                   # high, fixed view over the whole path
        cam.lookat[:] = [cx, cy, 0.0]; cam.distance = TOP_DIST; cam.azimuth = 90; cam.elevation = -65
    else:
        far = SLOPE or COURSE
        cam.lookat[:] = [q[0], q[1], q[2] - 0.1 if far else 0.15]
        cam.distance = 1.4 if not far else 2.4; cam.azimuth = 110 if not far else 95
        cam.elevation = -12 if not far else -6
        if COURSE:                    # side-on, so the hill, kicker and gap read
            cam.lookat[:] = [q[0], q[1], q[2] - 0.1]; cam.distance = 4.0; cam.azimuth = 90; cam.elevation = -10
    r.update_scene(d, cam, opt); img = r.render(); wr.append_data(img)
    if k in picks: frames.append(img.copy())
wr.close()
imageio.imwrite(out.replace(".mp4", "_sheet.png"), np.vstack([np.hstack(frames[:4]), np.hstack(frames[4:8])]))
print("ok", out)
