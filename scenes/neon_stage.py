"""A dark stage for glowing skins: deep navy grid floor, black-violet sky,
a dim cool key and two coloured rim lights (cyan and magenta).

Display only -- the ground geometry and friction are the default ones.
"""

from __future__ import annotations

from .registry import GROUND, Headlight, Scene


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.07, 0.07, 0.10), diffuse=(0.14, 0.14, 0.18)),
        textures=(
            sc.TextureCfg(
                name="neon_floor", type="2d", builtin="checker",
                rgb1=(0.035, 0.04, 0.08), rgb2=(0.06, 0.07, 0.13),
                width=512, height=512,
            ),
            sc.TextureCfg(
                name="neon_sky", type="skybox", builtin="gradient",
                rgb1=(0.08, 0.03, 0.14), rgb2=(0.0, 0.0, 0.02),
                width=256, height=256,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="neon_floor", texture="neon_floor", texuniform=True,
                texrepeat=(16.0, 16.0), reflectance=0.25, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="key", type="directional", pos=(-2.0, -2.0, 3.0),
                        dir=(0.5, 0.5, -1.0), diffuse=(0.45, 0.47, 0.55)),
            sc.LightCfg(name="rim_cyan", type="directional", pos=(3.0, 2.0, 1.5),
                        dir=(-0.7, -0.4, -0.6), diffuse=(0.0, 0.35, 0.45),
                        castshadow=False),
            sc.LightCfg(name="rim_magenta", type="directional", pos=(-3.0, 2.0, 1.5),
                        dir=(0.7, -0.4, -0.6), diffuse=(0.45, 0.05, 0.35),
                        castshadow=False),
        ),
    )
