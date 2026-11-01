"""PPO hyperparameters for jumper.cartwheel.

The jump's settings (same reference-tracking scheme, same observation width),
except **no left-right symmetry**: a cartwheel rolls one way, and its reference
observations (phase, future joint targets) describe that one direction. Mirroring
them would teach the policy a left-rolling state paired with a right-rolling
reference.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.cartwheel", symmetry=False)
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.algorithm.entropy_coef = 0.005
    cfg.actor.distribution_cfg["init_std"] = 0.6
    return cfg
