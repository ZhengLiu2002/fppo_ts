"""Student PPO runner configuration for Galileo CRL tasks."""

from __future__ import annotations

from isaaclab.utils import configclass

from ._shared_runner_cfg import build_algorithm_cfg, build_policy_cfg
from .rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg


@configclass
class GalileoCRLStudentPPORunnerCfg(CRLRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "galileo_fppo"
    empirical_normalization = False
    policy = build_policy_cfg("student")
    algorithm = build_algorithm_cfg("student")


__all__ = ["GalileoCRLStudentPPORunnerCfg"]
