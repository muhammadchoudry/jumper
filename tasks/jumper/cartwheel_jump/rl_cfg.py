"""PPO hyperparameters for jumper.cartwheel_jump: the cartwheel task's (no symmetry)."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.cartwheel_jump", symmetry=False)
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.algorithm.entropy_coef = 0.005
    cfg.actor.distribution_cfg["init_std"] = 0.6
    return cfg
