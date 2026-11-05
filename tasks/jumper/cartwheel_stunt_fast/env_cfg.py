"""jumper.cartwheel_stunt_fast: see tasks/jumper/cartwheel_stunt/env_cfg.py."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_stunt.env_cfg import build_stunt


def env_cfg(asset: Path | None = None, play: bool = False):
    return build_stunt(asset, play, "fast")


__all__ = ["env_cfg"]
