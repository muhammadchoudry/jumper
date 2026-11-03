"""jumper.cartwheel_eight: `cartwheel_full.env_cfg.build` on `ref/cartwheel_eight.npz`, steering-aware."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_full.env_cfg import build

REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_eight.npz"
#: 0.3 s stand + 10.8 s of reference + time to settle
EPISODE_S = 11.8


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = build(asset=asset, play=play, motion=REF_NPZ, steer=True, episode_s=EPISODE_S)
    if not play:
        # a quarter of the practice starts around the crossing, between the loops
        ev = cfg.events["reset_from_reference"]
        ev.params = {**ev.params, "focus_fraction": 0.2, "roll_window": (0.3, 0.7), "roll_fraction": 0.25}
    return cfg


__all__ = ["env_cfg", "REF_NPZ"]
