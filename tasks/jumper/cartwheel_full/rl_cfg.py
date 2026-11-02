"""PPO hyperparameters for jumper.cartwheel_full: jumper.cartwheel_roll's, so the
rolling policy's checkpoint loads as the starting point (same network)."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.cartwheel_full", symmetry=False)
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.algorithm.entropy_coef = 0.008
    cfg.actor.distribution_cfg["init_std"] = 0.5
    return cfg
