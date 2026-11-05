"""jumper.cartwheel_hill: `cartwheel_full.env_cfg.build` on cartwheel_ten's reference,
on an 8 deg downhill, holding the star (straight, evenly spread legs) while rolling."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_full.env_cfg import build
from ..cartwheel_ten.env_cfg import EPISODE_S, REF_NPZ

SLOPE_DEG = 8.0


def env_cfg(asset: Path | None = None, play: bool = False):
    return build(asset=asset, play=play, motion=REF_NPZ, steer=False, episode_s=EPISODE_S, straight_legs="even",
                 slope_deg=SLOPE_DEG, even_weight=5.0, even_sigma=0.25)


__all__ = ["env_cfg", "SLOPE_DEG"]
