"""Render a recorded rollout (roll_record.py's npz) to an mp4 + 8-frame contact sheet,
on a checker floor.

  MUJOCO_GL=egl python tools/crab_trio/traj_video.py roll_traj.npz out.mp4
"""
import sys
import numpy as np
import mujoco
import imageio
D = np.load(sys.argv[1]); out = sys.argv[2]
spec = mujoco.MjSpec.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "assets/jumper/jumper.xml"))
tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       rgb1=[0.12, 0.14, 0.2], rgb2=[0.2, 0.22, 0.3], width=300, height=300)
mat = spec.add_material(name="grid", texrepeat=[8, 8]); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="grid", contype=0, conaffinity=0)
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
m = spec.compile()
d = mujoco.MjData(m)
qa = [m.jnt_qposadr[m.joint(str(n)).id] for n in D["names"]]
r = mujoco.Renderer(m, 480, 854); cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
wr = imageio.get_writer(out, fps=30, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
dt = float(D["dt"]); step = max(1, int(round(1 / 30 / dt))); frames = []
for k in range(0, len(D["qs"]), step):
    q = D["qs"][k]; d.qpos[:] = 0; d.qpos[0:3] = q[0:3]; d.qpos[3:7] = q[3:7]; d.qpos[qa] = q[7:]
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], 0.15]; cam.distance = 1.4; cam.azimuth = 110; cam.elevation = -12
    r.update_scene(d, cam, opt); img = r.render(); wr.append_data(img)
    if len(frames) < 8 and k % (step * 8) == 0: frames.append(img.copy())
wr.close()
imageio.imwrite(out.replace(".mp4", "_sheet.png"), np.vstack([np.hstack(frames[:4]), np.hstack(frames[4:8])]))
print("ok", out)
