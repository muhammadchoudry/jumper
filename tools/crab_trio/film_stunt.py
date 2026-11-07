"""THE INFERNO GAP -- a cinematic cut of a recorded stunt run (full_record.py's npz).

Same physics as the recording: this only re-draws the run. It uses the fire
livery (tools/fire_skin.py), a desert mesa course with a fire pit in the gap,
sunset sky and haze, props and embers. It cuts between several cameras, with two
slow-motion angles of the jump, title cards, a colour grade and an original
synthesized score.

  MUJOCO_GL=egl python tools/crab_trio/film_stunt.py long_s0.npz out.mp4 [PREVIEW]
  (PREVIEW: 960x408 and no audio, for checking shots)
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys
import wave

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.jumper.cartwheel_stunt.course import LONG as C  # noqa: E402

NPZ, OUT = sys.argv[1], sys.argv[2]
PREVIEW = len(sys.argv) > 3
STILLS = __import__("os").environ.get("STILLS") == "1"     # one frame per shot, to check framing
W, H = (960, 408) if PREVIEW else (1920, 816)          # 2.35:1
FPS = 30
rng = np.random.default_rng(3)
D = np.load(NPZ)
QS, DT = D["qs"], float(D["dt"])
T_END = (len(QS) - 1) * DT
GROUND = C.RUNOUT_Z
Y0 = float(np.median(QS[int(2.5 / DT):int(7 / DT), 1]))  # the set is centred on the run's line
HW = 1.5                                                  # half width of the mesas (the run stays within ~1 m)
T_LIP = float(np.argmax(QS[:, 0] > C.LIP_X) * DT)
T_LAND = float(np.argmax(QS[:, 0] > C.LAND_X + 0.15) * DT)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
TMP = pathlib.Path(__import__("tempfile").mkdtemp(prefix="inferno_"))   # textures, the silent picture, the score


# --------------------------------------------------------------------------- textures
def _noise(h, w, scale, octaves=4, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(scale * 2 ** o))
        g = r.random((s + 1, s + 1))
        img = np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0
        out += amp * img
        tot += amp
        amp *= 0.5
    return out / tot


def make_textures():
    n = _noise(512, 512, 8, seed=1)
    sand = np.stack([0.62 * n + 0.30, 0.50 * n + 0.24, 0.40 * n + 0.18], -1)
    Image.fromarray((np.clip(sand, 0, 1) * 255).astype(np.uint8)).save(TMP / "sand.png")
    n = _noise(512, 512, 4, seed=2)
    strata = 0.5 + 0.5 * np.sin(np.linspace(0, 40, 512))[:, None] * 0.3 + 0.3 * n
    rock = np.stack([0.55 * strata + 0.15, 0.30 * strata + 0.08, 0.20 * strata + 0.05], -1)
    Image.fromarray((np.clip(rock, 0, 1) * 255).astype(np.uint8)).save(TMP / "rock.png")
    n = _noise(256, 256, 16, seed=3)
    grain = 0.5 + 0.5 * np.sin(np.linspace(0, 60, 256))[None, :] * 0.15 + 0.2 * n
    ply = np.stack([0.72 * grain + 0.15, 0.55 * grain + 0.10, 0.36 * grain + 0.06], -1)
    Image.fromarray((np.clip(ply, 0, 1) * 255).astype(np.uint8)).save(TMP / "ply.png")
    x = np.arange(256)
    stripes = ((x[:, None] + x[None, :]) // 32) % 2
    hz = np.where(stripes[..., None] == 1, [0.95, 0.75, 0.05], [0.08, 0.07, 0.07])
    Image.fromarray((hz * 255).astype(np.uint8)).save(TMP / "hazard.png")
    ck = ((x[:, None] // 32 + x[None, :] // 32) % 2)
    Image.fromarray((np.where(ck[..., None] == 1, [0.95, 0.95, 0.92], [0.05, 0.05, 0.05]) * 255).astype(np.uint8)
                    ).save(TMP / "checker.png")


# --------------------------------------------------------------------------- the set
FLAMES, EMBERS, TORCHES = [], [], []


def build():
    make_textures()
    spec = mujoco.MjSpec.from_file(str(ROOT / "assets/jumper/jumper_fire.xml"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1088
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.map.znear = 0.002
    spec.visual.map.zfar = 80
    spec.visual.map.fogstart = 5.0      # x stat.extent (1 m): haze only on the distant buttes
    spec.visual.map.fogend = 45.0
    spec.visual.rgba.fog = [0.62, 0.40, 0.42, 1.0]     # dusky rose
    spec.visual.rgba.haze = [0.95, 0.58, 0.40, 1.0]
    spec.visual.headlight.ambient = [0.12, 0.10, 0.12]
    spec.visual.headlight.diffuse = [0.15, 0.13, 0.15]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.stat.extent = 1.0

    def tex(name, file, rep=(1, 1), **mk):
        spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(TMP / file))
        m = spec.add_material(name=name, texrepeat=list(rep), **mk)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.05, 0.05, 0.20], rgb2=[0.98, 0.50, 0.25], width=512, height=3072)
    tex("sand", "sand.png", (30, 30), texuniform=True, specular=0.05)
    tex("rock", "rock.png", (2, 2), texuniform=True, specular=0.1)
    tex("ply", "ply.png", (2, 2), texuniform=True, specular=0.3, shininess=0.3)
    tex("hazard", "hazard.png", (3, 3), texuniform=True, specular=0.4)
    tex("checker", "checker.png", (4, 1), specular=0.2)
    for name, rgba, em in (("flame_core", [1.0, 0.92, 0.55, 0.9], 1.0), ("flame_mid", [1.0, 0.55, 0.08, 0.7], 1.0),
                           ("flame_out", [0.95, 0.20, 0.03, 0.45], 0.9), ("ember", [1.0, 0.55, 0.12, 1.0], 1.0),
                           ("coal", [0.35, 0.05, 0.02, 1.0], 0.6), ("barrel_red", [0.70, 0.06, 0.04, 1], 0.0),
                           ("steel", [0.55, 0.55, 0.58, 1], 0.0), ("pennant", [1.0, 0.75, 0.1, 1], 0.2),
                           ("butte", [0.26, 0.11, 0.10, 1], 0.0)):
        spec.add_material(name=name, rgba=rgba, emission=em, specular=0.3)
    wb = spec.worldbody
    g0 = dict(contype=0, conaffinity=0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60, 60, 0.1], pos=[10, Y0, GROUND - 0.08], material="sand", **g0)

    # mesas: a rock prism under every course segment, a plywood deck on top
    def prism(x1, z1, x2, z2, top_drop=0.02, mat="rock"):
        z1, z2 = z1 - top_drop, z2 - top_drop
        v = []
        for (x, z) in ((x1, z1), (x2, z2), (x2, GROUND - 0.05), (x1, GROUND - 0.05)):
            v += [[x, Y0 - HW, z], [x, Y0 + HW, z]]
        mesh = spec.add_mesh(name=f"mesa_{len(spec.meshes)}", uservert=np.array(v).flatten().tolist())
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, material=mat, **g0)

    def deck(x1, z1, x2, z2, mat):
        dx, dz = x2 - x1, z2 - z1
        L = math.hypot(dx, dz)
        nx, nz = -dz / L, dx / L
        t = 0.06
        phi = -math.atan2(dz, dx)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, HW, t / 2],
                    pos=[(x1 + x2) / 2 - nx * t / 2, Y0, (z1 + z2) / 2 - nz * t / 2],
                    quat=[math.cos(phi / 2), 0, math.sin(phi / 2), 0], material=mat, **g0)

    up = C.TAKEOFF
    for i, ((x1, z1), (x2, z2)) in enumerate(zip(up[:-1], up[1:])):
        prism(x1, z1, x2, z2)
        deck(x1, z1, x2, z2, "hazard" if i >= len(up) - 3 else "ply")
    dn = C.LANDING
    for i, ((x1, z1), (x2, z2)) in enumerate(zip(dn[:-1], dn[1:])):
        if z1 > GROUND + 0.01 or z2 > GROUND + 0.01:
            prism(x1, z1, x2, z2)
        deck(x1, z1, x2, z2, "hazard" if i == 0 else "ply")

    # the fire pit: flames on a bed of coals across the gap
    pit_x0, pit_x1 = C.LIP_X + 0.03, C.LAND_X - 0.03
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(pit_x1 - pit_x0) / 2, HW, 0.03],
                pos=[(pit_x0 + pit_x1) / 2, Y0, GROUND + 0.03], material="coal", **g0)
    for i in range(70):
        x = rng.uniform(pit_x0 + 0.05, pit_x1 - 0.05)
        y = Y0 + rng.uniform(-HW + 0.1, HW - 0.1)
        h = rng.uniform(0.25, 0.65)
        for layer, (mat, s) in enumerate((("flame_out", 1.0), ("flame_mid", 0.65), ("flame_core", 0.35))):
            g = wb.add_geom(name=f"flame_{i}_{layer}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                            size=[0.07 * s, 0.07 * s, h * s / 2], pos=[x, y, GROUND + h * s / 2], material=mat, **g0)
            FLAMES.append((g.name, x, y, h * s, 0.07 * s, rng.uniform(0, 2 * math.pi), rng.uniform(5, 11)))
    # torches on poles at the four lip corners
    for (x, z) in ((C.LIP_X - 0.15, C.LIP_Z), (C.LAND_X + 0.15, C.LAND_Z - 0.15)):
        for side in (-1, 1):
            y = Y0 + side * (HW - 0.12)
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.025, 0.45], pos=[x, y, z + 0.45], material="steel", **g0)
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.04], pos=[x, y, z + 0.92], material="steel", **g0)
            for layer, (mat, s) in enumerate((("flame_out", 1.0), ("flame_mid", 0.65), ("flame_core", 0.35))):
                g = wb.add_geom(name=f"torch_{len(TORCHES)}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                size=[0.08 * s, 0.08 * s, 0.17 * s], pos=[x, y, z + 0.96 + 0.17 * s], material=mat, **g0)
                TORCHES.append((g.name, x, y, z + 0.96, 0.34 * s, 0.08 * s, rng.uniform(0, 6), rng.uniform(7, 12)))
            wb.add_light(pos=[x, y, z + 1.2], dir=[0, 0, -1], diffuse=[0.6, 0.3, 0.08], attenuation=[0.2, 0.6, 1.2],
                         castshadow=False, type=mujoco.mjtLightType.mjLIGHT_POINT)
    # barrels along the kicker and the landing deck
    for (x, side) in ((C.LIP_X - 0.6, -1), (C.LIP_X - 0.6, 1), (C.LIP_X - 0.95, 1), (C.LAND_X + 1.4, -1),
                      (C.RUNOUT_X + 0.6, 1), (C.RUNOUT_X + 0.9, 1)):
        z = float(C.surface_z(__import__("torch").tensor([x]))[0])
        y = Y0 + side * (HW - 0.3)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.1, 0.14], pos=[x, y, z + 0.14], material="barrel_red", **g0)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.102, 0.012], pos=[x, y, z + 0.2], material="steel", **g0)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.102, 0.012], pos=[x, y, z + 0.08], material="steel", **g0)
    # pennants on a line over the start, and a start gate
    for side in (-1, 1):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.6], pos=[-0.4, Y0 + side * 1.0, 0.6], material="steel", **g0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 1.0, 0.07], pos=[-0.4, Y0, 1.15], material="checker", **g0)
    for i in range(14):
        y = Y0 - 1.3 + 2.6 * i / 13
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.06, 0.07],
                    pos=[C.LIP_X - 0.15, y, C.LIP_Z + 0.82 - 0.05 * math.cos(math.pi * (i / 13 - 0.5))],
                    material="pennant" if i % 2 else "barrel_red", **g0)
    # buttes on the horizon
    for (x, y, sx, sy, h) in ((30, 18, 4, 3, 5), (40, -14, 6, 4, 7), (22, -25, 3, 3, 4), (55, 6, 8, 5, 9), (-20, 22, 5, 4, 6)):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, h / 2], pos=[x, Y0 + y, GROUND + h / 2], material="butte", **g0)
    # embers that trail the crab
    for i in range(60):
        g = wb.add_geom(name=f"ember_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006], pos=[0, 0, -10],
                        material="ember", **g0)
        EMBERS.append(g.name)
    # lights: a low warm sun, a cool fill from the other side, and the pit's glow
    wb.add_light(pos=[-10, -8, 6], dir=[0.7, 0.55, -0.45], diffuse=[1.0, 0.72, 0.5], specular=[0.3, 0.25, 0.2],
                 castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(pos=[10, 10, 6], dir=[-0.5, -0.6, -0.6], diffuse=[0.18, 0.22, 0.38], castshadow=False,
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(name="pitglow", pos=[(pit_x0 + pit_x1) / 2, Y0, GROUND + 0.6], dir=[0, 0, 1], diffuse=[1.0, 0.45, 0.1],
                 attenuation=[0.1, 0.4, 0.4], castshadow=False, type=mujoco.mjtLightType.mjLIGHT_POINT)
    return spec


# --------------------------------------------------------------------------- shots
def hub(t):
    return pose(t)[0:3]


def pose(t):
    """Interpolated recorded state at time t (slow motion needs in-between frames)."""
    f = min(max(t / DT, 0), len(QS) - 1.001)
    i, a = int(f), f - int(f)
    q0, q1 = QS[i], QS[i + 1]
    q = (1 - a) * q0 + a * q1
    qa, qb = q0[3:7], q1[3:7].copy()
    if np.dot(qa, qb) < 0:
        qb = -qb
    qq = (1 - a) * qa + a * qb
    q[3:7] = qq / np.linalg.norm(qq)
    return q


def smooth_hub(t, w=0.15):
    ts = np.linspace(t - w, t + w, 9)
    return np.mean([hub(min(max(s, 0), T_END)) for s in ts], 0)


def ease(u):
    return u * u * (3 - 2 * u)


# each shot: (name, sim t0, sim t1, slow-motion factor, camera(t, u) -> (eye, target, fovy), caption)
def _establish(t, u):
    h = smooth_hub(t, 0.4)
    eye = np.array([-2.6 + 1.4 * ease(u), Y0 - 3.2 + 0.8 * ease(u), 1.6 - 0.5 * ease(u)])
    return eye, np.array([h[0] + 1.0, Y0, -0.2]), 42


def _edge(t, u):
    h = smooth_hub(t)
    return np.array([h[0] + 0.9, Y0 - 1.6, h[2] + 0.15]), h + [0, 0, 0.05], 38


def _chase(t, u):
    h = smooth_hub(t, 0.25)
    return np.array([h[0] - 1.25, h[1] - 0.35, h[2] + 0.45]), np.array([C.LIP_X + 0.6, Y0, C.LIP_Z]), 50


def _pit(t, u):        # from the bottom of the pit, looking up past the flames at the lip
    h = hub(t)
    eye = np.array([C.LIP_X + 0.4, Y0 - HW - 0.7, GROUND + 0.35])
    return eye, 0.6 * h + 0.4 * np.array([C.LIP_X + 0.35, Y0, C.LIP_Z + 0.2]), 40


def _front(t, u):      # from the landing deck, the crab coming over the pit at the camera
    h = hub(t)
    eye = np.array([C.LAND_X + 1.3, Y0 + 0.75, C.LAND_Z + 0.05])
    return eye, h, 28


def _landing(t, u):
    h = smooth_hub(t, 0.1)
    shake = 0.012 * math.exp(-max(t - T_LAND, 0) * 6) * np.sin(np.array([47, 61, 53]) * t) if t > T_LAND else 0
    return np.array([h[0] - 0.2, Y0 - 1.9, h[2] + 0.25]) + shake, h, 40


def _finale(t, u):
    h = smooth_hub(t, 0.4)
    eye = np.array([h[0] + 1.6 - 0.4 * ease(u), Y0 - 1.6 - 0.9 * ease(u), h[2] + 0.35 + 1.0 * ease(u)])
    return eye, np.array([h[0] - 0.8 * ease(u), Y0, h[2] - 0.1]), 45


SHOTS = [
    ("establish", 0.0, 2.4, 1.0, _establish, None),
    ("edge", 2.4, 3.6, 1.0, _edge, None),
    ("chase", 3.6, T_LIP - 0.12, 1.0, _chase, None),
    ("pit", T_LIP - 0.12, T_LAND + 0.15, 4.0, _pit, "4x SLOW MOTION"),
    ("front", T_LIP - 0.18, T_LAND + 0.1, 3.0, _front, "REPLAY"),
    ("landing", T_LAND - 0.05, min(T_LAND + 2.0, T_END), 1.0, _landing, None),
    ("finale", min(T_LAND + 2.0, T_END - 1.0), T_END, 1.0, _finale, None),
]


# --------------------------------------------------------------------------- render
def set_cam(cam, m, eye, target, fovy):
    v = np.asarray(target, float) - np.asarray(eye, float)
    dist = float(np.linalg.norm(v))
    cam.lookat[:] = target
    cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
    cam.elevation = math.degrees(math.asin(v[2] / dist))
    m.vis.global_.fovy = fovy


def animate(m, d, t, wall):
    """Flicker the flames and torches, move the embers (wall = film time, so the
    fire keeps moving in slow motion, just slower)."""
    for name, x, y, h, r, ph, f in FLAMES:
        gid = m.geom(name).id
        k = 1 + 0.25 * math.sin(f * wall + ph) + 0.12 * math.sin(2.3 * f * wall + 2 * ph)
        m.geom_size[gid] = [r * (1.1 - 0.2 * k + 0.2), r * (1.1 - 0.2 * k + 0.2), h * k / 2]
        m.geom_pos[gid] = [x + 0.015 * math.sin(3 * wall + ph), y, GROUND + h * k / 2 + 0.02]
    for name, x, y, z, h, r, ph, f in TORCHES:
        gid = m.geom(name).id
        k = 1 + 0.2 * math.sin(f * wall + ph) + 0.1 * math.sin(2.7 * f * wall)
        m.geom_size[gid] = [r, r, h * k / 2]
        m.geom_pos[gid] = [x, y, z + h * k / 2]
    li = m.light("pitglow").id
    m.light_diffuse[li] = np.array([1.0, 0.45, 0.1]) * (0.85 + 0.15 * math.sin(13 * wall) * math.sin(7.1 * wall))
    # embers: born at the crab every few ms of sim time while it rolls, then rise and fade
    for i, name in enumerate(EMBERS):
        gid = m.geom(name).id
        life = 0.7
        born = math.floor((t - i * life / len(EMBERS)) / life) * life + i * life / len(EMBERS)
        age = t - born
        if born < 2.6 or born > T_END - 0.5:
            m.geom_pos[gid] = [0, 0, -50]
            continue
        p0 = hub(born)
        seed = (i * 7919 + int(born * 1000)) % 1000 / 1000
        drift = np.array([-0.35 * age, 0.25 * math.sin(40 * seed) * age, 0.55 * age - 0.3 * age * age])
        m.geom_pos[gid] = p0 + np.array([0.12 * math.cos(30 * seed), 0.12 * math.sin(50 * seed), 0.05]) + drift
        m.geom_size[gid, 0] = 0.005 * max(0.0, 1 - age / life) + 0.0008


def caption(img, text, size, y, alpha=1.0, color=(255, 236, 200), font=FONT_B, glow=True, spacing=None):
    pil = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    f = ImageFont.truetype(font, int(size * W / 1920))
    bbox = dr.textbbox((0, 0), text, font=f)
    x = (W - (bbox[2] - bbox[0])) / 2
    yy = int(y * H)
    if glow:
        gl = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).text((x, yy), text, font=f, fill=(255, 110, 20, int(200 * alpha)))
        layer = Image.alpha_composite(layer, gl.filter(ImageFilter.GaussianBlur(10 * W / 1920)))
        dr = ImageDraw.Draw(layer)
    dr.text((x, yy), text, font=f, fill=color + (int(255 * alpha),))
    return np.array(Image.alpha_composite(pil, layer).convert("RGB"))


def corner(img, text):
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT_B, int(26 * W / 1920))
    dr.text((int(0.035 * W), int(0.06 * H)), text, font=f, fill=(255, 220, 170))
    return np.array(pil)


def main():
    spec = build()
    m = spec.compile()
    d = mujoco.MjData(m)
    qa = [m.jnt_qposadr[m.joint(str(n)).id] for n in D["names"]]
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    raw = TMP / "picture.mp4"
    wr = imageio.get_writer(str(raw), fps=FPS, codec="libx264", quality=9, macro_block_size=8, pixelformat="yuv420p")
    wall = 0.0
    marks = {}
    sheet = []

    def frame(t, eye, target, fovy, cap=None):
        nonlocal wall
        q = pose(t)
        d.qpos[:] = 0
        d.qpos[0:3], d.qpos[3:7], d.qpos[qa] = q[0:3], q[3:7], q[7:]
        animate(m, d, t, wall)
        mujoco.mj_forward(m, d)
        set_cam(cam, m, eye, target, fovy)
        r.update_scene(d, cam, opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0     # fog would also paint over the sky
        r.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        img = r.render()
        if cap:
            img = corner(img, cap)
        wall += 1 / FPS
        return img

    # opening title over the establishing shot
    for name, t0, t1, slow, fn, cap in SHOTS:
        marks[name] = wall
        n = max(1, int(round((t1 - t0) * slow * FPS)))
        for k in (range(n) if not STILLS else [n // 2]):
            u = k / max(1, n - 1)
            t = t0 + (t1 - t0) * k / n
            eye, target, fovy = fn(t, u)
            img = frame(t, eye, target, fovy, cap)
            if name == "establish":
                a = min(1, k / 12) * min(1, (n - k) / 12)
                img = caption(img, "THE INFERNO GAP", 120, 0.30, a)
                img = caption(img, "0.6 m robot crab  ·  0.75 m gap  ·  simulated physics", 34, 0.52, a,
                              color=(255, 235, 215), font=FONT_R, glow=False)
            if k == n // 2:
                sheet.append(img)
            wr.append_data(img)
        print(f"shot {name}: {n} frames", flush=True)
    if STILLS:
        wr.close()
        Image.fromarray(np.vstack([np.hstack(sheet[:4]), np.hstack((sheet[4:] + [np.zeros_like(sheet[0])] * 4)[:4])])
                        ).save(str(OUT).replace(".mp4", "_stills.png"))
        return
    # end card over a held, darkened last frame
    last = img
    for k in range(int(3.0 * FPS)):
        a = min(1, k / 15)
        img = (last * (1 - 0.55 * a)).astype(np.uint8)
        img = caption(img, "LANDED.", 110, 0.24, a)
        img = caption(img, "launch 2.8 m/s   ·   0.75 m over the fire   ·   23 cm to spare", 34, 0.48, a,
                      color=(255, 235, 215), font=FONT_R, glow=False)
        img = caption(img, "trained with reinforcement learning  ·  same physics as the training simulator", 26, 0.60, a,
                      color=(230, 200, 180), font=FONT_R, glow=False)
        wr.append_data(img)
    marks["end"] = wall
    wr.close()
    sheet = (sheet + [np.zeros_like(sheet[0])] * 8)[:8]
    Image.fromarray(np.vstack([np.hstack(sheet[:4]), np.hstack(sheet[4:8])])).resize((W * 2, H)).save(
        str(OUT).replace(".mp4", "_sheet.png"))
    total = wall
    launch = marks["pit"] + 0.12 * 4.0
    print("marks", {k: round(v, 2) for k, v in marks.items()}, "total", round(total, 2))
    grade = "eq=contrast=1.08:saturation=1.18:gamma=0.97,vignette=PI/5"
    if PREVIEW:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", grade, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", OUT], check=True)
    else:
        wav = TMP / "score.wav"
        score(wav, total, marks, launch)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(wav), "-vf", grade,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", OUT], check=True)
    print("ok", OUT)


# --------------------------------------------------------------------------- score
SR = 44100


def score(path, total, marks, launch):
    """An original cue: a low drone and pulse that build to the launch, a riser, a hit
    and a held breath through the slow motion (a heartbeat and the fire), a boom on
    the landing, and a resolving chord on the end card. Fire crackle throughout."""
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)

    def hz(mid):
        return 440 * 2 ** ((mid - 69) / 12)

    def add(sig, at):
        i = int(at * SR)
        j = min(n, i + len(sig))
        if j > i:
            out[i:j] += sig[: j - i]

    slow0, slow1 = marks["pit"], marks["landing"]
    # drone (D minor) with a slow swell, ducked during the slow motion
    drone = sum(np.sin(2 * np.pi * hz(m) * t + k) for k, m in enumerate((38, 45, 50, 53)))
    swell = np.clip(t / max(launch, 1), 0, 1) ** 1.5
    duck = 1 - 0.75 * ((t > slow0) & (t < slow1))
    out += 0.05 * drone * (0.4 + 0.6 * swell) * duck
    # driving pulse up to the launch
    beat = 60 / 132
    for b in np.arange(0.0, launch - 0.1, beat / 2):
        k = np.arange(int(0.12 * SR)) / SR
        add(0.25 * np.sin(2 * np.pi * (55 + 40 * np.exp(-30 * k)) * k) * np.exp(-20 * k) * (0.5 + 0.5 * b / launch), b)
        if int(b / beat * 2) % 4 == 2:
            add(0.06 * rng.standard_normal(len(k)) * np.exp(-35 * k), b)
    # riser into the launch
    rl = 2.0
    k = np.arange(int(rl * SR)) / SR
    noise = rng.standard_normal(len(k))
    add(0.12 * noise * (k / rl) ** 3 + 0.05 * np.sin(2 * np.pi * (200 + 900 * (k / rl) ** 2) * k) * (k / rl) ** 2,
        launch - rl)
    # hit at the launch, then the held breath: heartbeat + fire roar
    k = np.arange(int(1.5 * SR)) / SR
    add(0.5 * np.sin(2 * np.pi * (45 + 60 * np.exp(-12 * k)) * k) * np.exp(-3 * k), launch)
    for hb in np.arange(launch + 0.4, slow1, 0.9):
        kk = np.arange(int(0.25 * SR)) / SR
        thump = np.sin(2 * np.pi * 50 * kk) * np.exp(-25 * kk)
        add(0.35 * thump, hb)
        add(0.22 * thump, hb + 0.22)
    # landing boom (the replay ends just after it, the landing shot starts with it)
    k = np.arange(int(2.0 * SR)) / SR
    add(0.7 * np.sin(2 * np.pi * (38 + 80 * np.exp(-15 * k)) * k) * np.exp(-2.5 * k)
        + 0.15 * rng.standard_normal(len(k)) * np.exp(-8 * k), slow1 + 0.05)
    # resolving chord on the end card (D major)
    endc = marks["finale"]
    chord = sum(np.sin(2 * np.pi * hz(m) * t) for m in (50, 54, 57, 62, 66))
    out += 0.05 * chord * np.clip((t - endc) / 1.5, 0, 1) * np.clip((total - t) / 1.0, 0, 1)
    # fire crackle and roar
    roar = np.convolve(rng.standard_normal(n), np.ones(60) / 60, "same")
    crackle = (rng.random(n) > 0.9995) * rng.standard_normal(n) * 3
    crackle = np.convolve(crackle, np.exp(-np.arange(200) / 30), "same")
    fire_level = 0.4 + 0.6 * ((t > slow0) & (t < slow1))
    out += (0.25 * roar + 0.05 * crackle) * fire_level
    out = np.tanh(1.3 * out) * 0.8
    fade = np.clip((total - t) / 0.8, 0, 1) * np.clip(t / 0.3, 0, 1)
    out *= fade
    st = np.stack([out, np.roll(out, 220)], 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
