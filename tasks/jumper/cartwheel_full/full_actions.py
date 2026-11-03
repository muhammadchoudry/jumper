"""Residual actions whose reach depends on the phase.

The roll needs room (0.7 rad around the star: stepping, folding, pumping); the
run-up, jump and stand need the jump task's tighter residual (0.4 rad, the scale
jumper.cartwheel_runup learned the trot with). With one scale of 0.7 everywhere,
the warm-started rolling policy's large outputs threw the trot more than 6 rad^2 off
the reference in its first second, every time, for 900 rounds (run
2026-11-02_15-04-33) -- and scaling alone was not enough: its raw outputs there are
several times the unit range, so outside the roll they are also capped at +-1.
"""

from __future__ import annotations

import torch

from ..jump.mdp.actions import (
    ReferenceResidualJointPositionAction,
    ReferenceResidualJointPositionActionCfg,
)
from . import full_mdp as F


class PhaseScaledResidualActionCfg(ReferenceResidualJointPositionActionCfg):
    other_phase_factor: float = 0.4 / 0.7
    """Multiplier on the residual outside phase B (scale is set for the roll)."""

    def build(self, env) -> "PhaseScaledResidualAction":
        return PhaseScaledResidualAction(self, env)


class PhaseScaledResidualAction(ReferenceResidualJointPositionAction):
    cfg: PhaseScaledResidualActionCfg

    def apply_actions(self) -> None:
        in_roll = F.phase_mask(self._env, "B").unsqueeze(1)
        # outside the roll: the raw output capped at +-1, times 0.4 rad
        capped = (self._raw_actions.clamp(-1.0, 1.0) * self._scale + self._offset) * self.cfg.other_phase_factor
        residual = torch.where(in_roll, self._processed_actions, capped)
        ref_q = self._get_ref_q()
        target = ref_q[:, self._target_ids] + residual
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)


__all__ = ["PhaseScaledResidualAction", "PhaseScaledResidualActionCfg"]
