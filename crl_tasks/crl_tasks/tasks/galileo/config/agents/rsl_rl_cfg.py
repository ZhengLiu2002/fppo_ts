"""RSL-RL policy and algorithm configuration for Galileo CRL tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CRLRslRlBaseCfg:
    """Shared observation layout for FPPO-style policies."""

    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    # + joint_pos(12) + joint_vel(12) + actions(12) + commands(3)
    # + height_scan(132) = 180
    num_priv_hurdles: int = 0
    num_priv_explicit: int = 0
    num_priv_latent: int = 0
    num_prop: int = 180
    num_scan: int = 0
    num_hist: int = 0


@configclass
class CRLRslRlStateHistEncoderCfg(CRLRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder"
    channel_size: int = 10


@configclass
class CRLRslRlActorCfg(CRLRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: CRLRslRlStateHistEncoderCfg = MISSING


@configclass
class CRLRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticRMA"
    num_prop: int = MISSING
    num_scan: int = 0
    num_priv_explicit: int = 0
    num_priv_latent: int = 0
    num_hist: int = 0
    tanh_encoder_output: bool = False
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    cost_critic_hidden_dims: list[int] | None = None
    encode_scan_for_critic: bool = False
    critic_scan_encoder_dims: list[int] | None = None
    critic_num_prop: int | None = None
    critic_num_scan: int | None = None
    critic_num_priv_explicit: int | None = None
    critic_num_priv_latent: int | None = None
    critic_num_hist: int | None = None
    actor: CRLRslRlActorCfg = MISSING


@configclass
class CRLRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "FPPO"
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float] = [0, 0.1, 2000, 3000]

    # FPPO/CMDP extensions
    cost_value_loss_coef: float = 1.0
    step_size: float = 1e-3
    cost_gamma: float | None = None
    cost_lam: float | None = None
    cost_limit: float = 0.0
    delta_safe: float | None = 0.01
    backtrack_coeff: float = 0.5
    max_backtracks: int = 10
    projection_eps: float = 1e-8
    use_clipped_surrogate: bool = True
    normalize_cost_advantage: bool = False
    constraint_normalization: bool = True
    constraint_norm_beta: float = 0.99
    constraint_norm_min_scale: float = 1e-3
    constraint_norm_max_scale: float = 10.0
    constraint_norm_clip: float = 5.0
    constraint_proxy_delta: float = 0.1
    constraint_agg_tau: float = 0.5
    constraint_scale_by_gamma: bool = False
    constraint_cost_scale: float | None = None
    use_preconditioner: bool = True
    preconditioner_beta: float = 0.999
    preconditioner_eps: float = 1e-8
    feasible_first: bool = True
    feasible_first_coef: float = 1.0
    # NP3O-style positive-part cost shaping (shared across constrained algorithms)
    cost_viol_loss_coef: float = 0.0
    k_value: float = 1.0
    k_growth: float = 1.0
    k_max: float = 1.0


@configclass
class CRLRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "Distillation"


@configclass
class CRLRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: CRLRslRlPpoActorCriticCfg = MISSING
    algorithm: CRLRslRlPpoAlgorithmCfg | CRLRslRlDistillationAlgorithmCfg = MISSING


__all__ = [
    "CRLRslRlBaseCfg",
    "CRLRslRlStateHistEncoderCfg",
    "CRLRslRlActorCfg",
    "CRLRslRlPpoActorCriticCfg",
    "CRLRslRlPpoAlgorithmCfg",
    "CRLRslRlDistillationAlgorithmCfg",
    "CRLRslRlOnPolicyRunnerCfg",
]
