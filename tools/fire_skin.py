"""Inferno: a display-only fire livery for Jumper (like tools/neon_reef.py).

Writes assets/jumper/jumper_fire.xml from assets/jumper/jumper.xml, plus
assets/jumper/fire/: the visual meshes re-saved as OBJ with texture coordinates,
and the flame textures. Only visual geoms (class="visual") change. Collision
geoms, masses, joints and limits are untouched, so physics is identical.

- shell: hot-rod flames. A solid yellow-orange-red front, with tongues streaming
  back over glossy black (planar map from above, u from front to back).
- limbs: flames licking out from each joint toward the tip (u along the part's
  long axis, from the end nearest its joint).
- pads, tips and grip inserts: glowing ember orange.
- base: charcoal; the display module, camera and sensor keep their look.

  python tools/fire_skin.py
"""
from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/jumper/jumper.xml"
DST = ROOT / "assets/jumper/jumper_fire.xml"
OUT = ROOT / "assets/jumper/fire"
LIMBS = ("LF_", "RF_", "LM_", "RM_", "LR_", "RR_")
rng = np.random.default_rng(11)


def _smooth_noise(n, k, octaves=3):
    """1-D smooth noise in [0, 1] over n samples (k bumps at the coarsest octave)."""
    out = np.zeros(n)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        pts = rng.random(k * 2 ** o + 2)
        out += amp * np.interp(np.linspace(0, len(pts) - 1, n), np.arange(len(pts)), pts)
        tot += amp
        amp *= 0.5
    return out / tot


def flame_texture(size=1024, tongues=7, reach=(0.35, 0.95), body=0.18, seed_shift=0):
    """RGB flames: solid fire for u < body, tongues of varying length out to `reach`,
    glossy black beyond. u runs along x (columns), v along y (rows)."""
    u = np.linspace(0, 1, size)[None, :]
    v = np.linspace(0, 1, size)[:, None]
    lo, hi = reach
    wob = _smooth_noise(size, 6)[:, None]
    tongue = np.abs(np.sin(np.pi * tongues * (v + 0.04 * wob + seed_shift)))
    length = body + (lo - body) + (hi - lo) * tongue ** 1.6 * (0.6 + 0.4 * _smooth_noise(size, 9)[:, None])
    # curl the tongues a little (a wave along u)
    length = length + 0.03 * np.sin(2 * np.pi * (3 * u + 2 * v))
    t = np.clip(u / length, 0, 1.2)                       # 0 at the root of the flame, 1 at its edge
    inside = t < 1.0
    # palette: white-yellow core -> yellow -> orange -> red edge, a thin dark-red outline
    stops = np.array([[0.0, 1.00, 0.97, 0.75], [0.25, 1.00, 0.86, 0.20], [0.55, 1.00, 0.50, 0.04],
                      [0.85, 0.88, 0.12, 0.02], [1.00, 0.45, 0.03, 0.02]])
    rgb = np.stack([np.interp(t, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], -1)
    black = np.array([0.03, 0.025, 0.03])
    edge = (t >= 0.96) & (t < 1.0)
    rgb[edge] = [0.25, 0.02, 0.01]
    img = np.where(inside[..., None], rgb, black)
    # faint ember speckle on the black
    speck = (rng.random((size, size)) > 0.9985) & ~inside
    img[speck] = [1.0, 0.45, 0.05]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def uv_along(mesh: trimesh.Trimesh, axis_u: np.ndarray | None = None, up: np.ndarray | None = None):
    """Planar UVs: u along axis_u (default: the long PCA axis, pointing away from the
    origin -- the joint), v along the next axis (or across `up`)."""
    V = mesh.vertices
    c = V.mean(0)
    if axis_u is None:
        _, _, vt = np.linalg.svd(V - c, full_matrices=False)
        a, b = vt[0], vt[1]
        if np.dot(c, a) < 0:                               # u grows away from the joint at the origin
            a = -a
    else:
        a = axis_u / np.linalg.norm(axis_u)
        b = np.cross(up, a)
        b /= np.linalg.norm(b)
    pu, pv = V @ a, V @ b
    u = (pu - pu.min()) / (np.ptp(pu) + 1e-9)
    v = (pv - pv.min()) / (np.ptp(pv) + 1e-9)
    return np.stack([u, v], 1)


def kind(name: str) -> str | None:
    name = name.removesuffix("_visual")
    if name == "upper_shell_link":
        return "shell"
    if name == "base_link":
        return "char"
    if name.startswith(LIMBS):
        if re.search(r"pad_|tip_|insert_", name):
            return "ember"
        return "limb"
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flame_texture(tongues=6, reach=(0.45, 0.98), body=0.22).save(OUT / "flames_shell.png")
    flame_texture(tongues=3, reach=(0.40, 0.95), body=0.15, seed_shift=0.3).save(OUT / "flames_limb.png")

    tree = ET.parse(SRC)
    root = tree.getroot()
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="fire_shell_tex", type="2d", file="fire/flames_shell.png")
    ET.SubElement(asset, "texture", name="fire_limb_tex", type="2d", file="fire/flames_limb.png")
    mats = {
        "fire_shell": dict(texture="fire_shell_tex", rgba="1 1 1 1", emission="0.35", specular="0.9", shininess="0.9",
                           reflectance="0.05"),
        "fire_limb": dict(texture="fire_limb_tex", rgba="1 1 1 1", emission="0.3", specular="0.7", shininess="0.8"),
        "fire_char": dict(rgba="0.06 0.05 0.05 1", specular="0.5", shininess="0.6"),
        "fire_ember": dict(rgba="1.0 0.42 0.05 1", emission="0.9", specular="0.3", shininess="0.4"),
    }
    for n, kw in mats.items():
        ET.SubElement(asset, "material", name=n, **kw)

    meshes = {m.get("name"): m for m in asset.iter("mesh")}
    redone = set()
    changed = 0
    for geom in root.iter("geom"):
        if geom.get("class") != "visual":
            continue
        k = kind(geom.get("name", ""))
        if not k:
            continue
        geom.attrib.pop("rgba", None)
        geom.set("material", f"fire_{k}")
        changed += 1
        mname = geom.get("mesh")
        if k in ("shell", "limb") and mname and mname not in redone:
            m = meshes[mname]
            src = ROOT / "assets/jumper" / m.get("file")
            tm = trimesh.load(src, force="mesh")
            if k == "shell":   # top view: u from the front (+x) back, v across
                uv = uv_along(tm, axis_u=np.array([-1.0, 0, 0]), up=np.array([0, 0, 1.0]))
            else:
                uv = uv_along(tm)
            tm.visual = trimesh.visual.TextureVisuals(uv=uv)
            obj = OUT / f"{mname}.obj"
            text = trimesh.exchange.obj.export_obj(tm, include_texture=True, include_normals=True,
                                                   write_texture=False, mtl_name=None)
            obj.write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith(("mtllib", "usemtl"))) + "\n")
            m.set("file", f"fire/{mname}.obj")
            redone.add(mname)
    root.set("model", "jumper_fire")
    tree.write(DST, encoding="unicode")
    print(f"wrote {DST.relative_to(ROOT)}: {changed} visual geoms, {len(redone)} meshes re-mapped")


if __name__ == "__main__":
    main()
