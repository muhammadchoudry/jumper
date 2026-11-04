"""jumper.cartwheel_ten_straight: `cartwheel_full.env_cfg.build` on cartwheel_ten's
reference, with straight legs rewarded while rolling."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_full.env_cfg import build
from ..cartwheel_ten.env_cfg import EPISODE_S, REF_NPZ


def env_cfg(asset: Path | None = None, play: bool = False):
    return build(asset=asset, play=play, motion=REF_NPZ, steer=False, episode_s=EPISODE_S, straight_legs=True)


__all__ = ["env_cfg"]
