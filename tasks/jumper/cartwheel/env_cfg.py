"""jumper.cartwheel -- reference-guided sideways cartwheel.

Built on `jumper.jump`'s reference-tracking scheme (200 Hz control, kp 20,
residual actions around the reference, reference-state initialisation, dense
imitation rewards), with what a cartwheel needs changed:

- **Reference**: `ref/cartwheel_flat.npz` (kinematic, authored by
  `tools/crab_trio/cartwheel_ref.py`): fanned crouch -> sideways push -> limbs
  straight into a six-spoke star -> 360 degree roll -> stance -> home. Straight
  limbs carry load along their length, which is what keeps the spread pose inside
  the 1.75 N*m motors.
- **Spawn keeps the spin** (`mdp/events.py`): an env born upside down mid-roll
  gets the reference's roll rate, not zero.
- **No `fell_over`**: a cartwheel is upside down on purpose. An episode ends when
  orientation strays > 1 rad from the reference's (`mdp/terminations.py`) or the
  joints diverge.
- **Claws are feet here**: the claw tips carry the body over the top, so they are
  not undesired contacts; only chassis and upper-limb segments are. The jump's
  foot-contact schedule reward is dropped -- the claw tips are not the jump's
  foot bodies, and the tracking terms already pin the motion.

The reference is kinematic, not a physics recording, so a perfect imitation
score may not be reachable; the policy finds the closest physical motion.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..common.constants import CHASSIS_LINKS, LIMB_LINKS
from ..jump.env_cfg import env_cfg as _jump_env_cfg
from ..jump.env_cfg import _FUTURE_OFFSETS, _SIM_TIMESTEP, _DECIMATION
from . import mdp as cw_mdp

CARTWHEEL_REF_NPZ = Path(__file__).resolve().parent / "ref" / "cartwheel_flat.npz"

#: Undesired ground contact: chassis and upper limbs only (claw and foot tips may land).
_UNDESIRED_GEOMS = tuple(f"{link}_meshcol" for link in CHASSIS_LINKS + LIMB_LINKS)


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = _jump_env_cfg(asset=asset, play=play)

    cfg.commands["jump"].motion_file = str(CARTWHEEL_REF_NPZ)

    sensors = []
    for s in cfg.scene.sensors or ():
        if s.name == "body_ground_contact":
            s = ContactSensorCfg(
                name="body_ground_contact",
                primary=ContactMatch(mode="geom", pattern=_UNDESIRED_GEOMS, entity="robot"),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("force",),
                reduce="netforce",
                num_slots=1,
            )
        sensors.append(s)
    cfg.scene.sensors = tuple(sensors)

    cfg.rewards.pop("ref_contact", None)

    cfg.terminations.pop("fell_over", None)
    cfg.terminations["orientation_diverged"] = TerminationTermCfg(
        func=cw_mdp.terminations.orientation_diverged,
        params={"command_name": "jump", "max_angle": 1.0},
    )

    cfg.events["reset_from_reference"].func = cw_mdp.events.reset_from_reference_phase

    # span 2.4 s after go plus a settling margin
    cfg.episode_length_s = 3.0
    cfg.viewer.distance = 1.6

    def _reference_contract() -> dict:
        from ..common.constants import HOME
        ref = cw_mdp.reference.JumpReference(CARTWHEEL_REF_NPZ, "cpu", list(HOME))
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=True)

    cfg.reference_contract = _reference_contract
    return cfg


__all__ = ["env_cfg", "CARTWHEEL_REF_NPZ"]
