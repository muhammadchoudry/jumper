"""jumper.cartwheel_eight: `cartwheel_full.env_cfg.build` on `ref/cartwheel_eight.npz`, steering-aware."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_full.env_cfg import build

#: try 3: a straight reference of the 8's roll length (spawn states, phases, and the
#: run-up / jump / stand to copy); the 8 itself is drawn by distance rolled
#: (full_mdp.eight_state), so the reference's own heading plays no part
REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_long.npz"
EIGHT_REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_eight.npz"   # tries 1-2
#: 0.3 s stand + 10.8 s of reference + time to settle
EPISODE_S = 11.8


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = build(asset=asset, play=play, motion=REF_NPZ, steer="distance", episode_s=EPISODE_S)
    if not play:
        # a quarter of the practice starts mid-roll: those begin at a random point along
        # their 8 (full_mdp.eight_state), so the crossing and both loops get practice
        ev = cfg.events["reset_from_reference"]
        ev.params = {**ev.params, "focus_fraction": 0.2, "roll_window": (0.1, 0.8), "roll_fraction": 0.25}
    return cfg


__all__ = ["env_cfg", "REF_NPZ"]
