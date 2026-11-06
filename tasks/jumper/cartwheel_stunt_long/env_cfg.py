"""jumper.cartwheel_stunt_long: the roll-faster stunt over a longer gap -- see
tasks/jumper/cartwheel_stunt/env_cfg.py and course.LONG."""

from __future__ import annotations

from pathlib import Path

from ..cartwheel_stunt.env_cfg import build_stunt


def env_cfg(asset: Path | None = None, play: bool = False):
    return build_stunt(asset, play, "fast", course_name="long", centre_weight=1.0)


__all__ = ["env_cfg"]
