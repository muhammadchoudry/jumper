"""jumper.cartwheel_ten: `cartwheel_full.env_cfg.build` on `ref/cartwheel_ten.npz`."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_full.env_cfg import build

REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_ten.npz"
#: 0.3 s stand + 9.1 s of reference + time to settle
EPISODE_S = 9.8


def env_cfg(asset: Path | None = None, play: bool = False):
    return build(asset=asset, play=play, motion=REF_NPZ, steer=False, episode_s=EPISODE_S)


__all__ = ["env_cfg", "REF_NPZ"]
