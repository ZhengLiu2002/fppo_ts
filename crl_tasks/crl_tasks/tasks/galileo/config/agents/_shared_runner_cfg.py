"""Shared helpers for Galileo RSL-RL runner configs."""

from __future__ import annotations

from typing import Literal

from ..defaults import GalileoDefaults
from .rsl_rl_cfg import (
    CRLRslRlActorCfg,
    CRLRslRlPpoActorCriticCfg,
    CRLRslRlPpoAlgorithmCfg,
    CRLRslRlStateHistEncoderCfg,
)

RunnerRole = Literal["teacher", "student"]

_ACTOR_HIDDEN_DIMS = [512, 512, 256, 128]
_CRITIC_HIDDEN_DIMS = [512, 512, 256, 128]
_SCAN_ENCODER_DIMS = [128, 64, 32]


def build_algorithm_cfg(role: RunnerRole) -> CRLRslRlPpoAlgorithmCfg:
    """Build algorithm config from ``GalileoDefaults`` for the given role."""
    algo_key = getattr(GalileoDefaults.algorithm, "name", "fppo")
    class_name = GalileoDefaults.algorithm.class_name_map.get(algo_key, algo_key)
    params: dict = {}
    params.update(getattr(GalileoDefaults.algorithm, "base", {}))
    params.update(getattr(GalileoDefaults.algorithm, "per_algo", {}).get(algo_key, {}))
    if role == "teacher":
        params.update(getattr(GalileoDefaults.algorithm, "teacher_override", {}))
    else:
        params.update(getattr(GalileoDefaults.algorithm, "student_override", {}))
    return CRLRslRlPpoAlgorithmCfg(class_name=class_name, **params)


def build_obs_cfg(role: RunnerRole) -> tuple[dict[str, int], dict[str, int]]:
    """Build actor/critic observation layout from ``GalileoDefaults``."""
    obs_cfg = getattr(GalileoDefaults.obs, role)
    actor_cfg = {
        "num_prop": obs_cfg.actor_num_prop,
        "num_scan": obs_cfg.actor_num_scan,
        "num_priv_explicit": obs_cfg.actor_num_priv_explicit,
        "num_priv_latent": obs_cfg.actor_num_priv_latent,
        "num_hist": obs_cfg.actor_num_hist,
    }
    critic_cfg = {
        "critic_num_prop": obs_cfg.critic_num_prop,
        "critic_num_scan": obs_cfg.critic_num_scan,
        "critic_num_priv_explicit": obs_cfg.critic_num_priv_explicit,
        "critic_num_priv_latent": obs_cfg.critic_num_priv_latent,
        "critic_num_hist": obs_cfg.critic_num_hist,
    }
    return actor_cfg, critic_cfg


def build_policy_cfg(role: RunnerRole) -> CRLRslRlPpoActorCriticCfg:
    """Build policy config from shared defaults and observation layout."""
    actor_obs, critic_obs = build_obs_cfg(role)
    return CRLRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        num_prop=actor_obs["num_prop"],
        num_scan=actor_obs["num_scan"],
        num_priv_explicit=actor_obs["num_priv_explicit"],
        num_priv_latent=actor_obs["num_priv_latent"],
        num_hist=actor_obs["num_hist"],
        encode_scan_for_critic=(critic_obs["critic_num_scan"] > 0),
        critic_num_prop=critic_obs["critic_num_prop"],
        critic_num_scan=critic_obs["critic_num_scan"],
        critic_num_priv_explicit=critic_obs["critic_num_priv_explicit"],
        critic_num_priv_latent=critic_obs["critic_num_priv_latent"],
        critic_num_hist=critic_obs["critic_num_hist"],
        critic_scan_encoder_dims=_SCAN_ENCODER_DIMS if critic_obs["critic_num_scan"] > 0 else None,
        actor_hidden_dims=_ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=_CRITIC_HIDDEN_DIMS,
        scan_encoder_dims=_SCAN_ENCODER_DIMS,
        priv_encoder_dims=[],
        activation="elu",
        actor=CRLRslRlActorCfg(
            class_name="Actor",
            num_prop=actor_obs["num_prop"],
            num_scan=actor_obs["num_scan"],
            num_priv_explicit=actor_obs["num_priv_explicit"],
            num_priv_latent=actor_obs["num_priv_latent"],
            num_hist=actor_obs["num_hist"],
            state_history_encoder=CRLRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
                num_prop=actor_obs["num_prop"],
                num_hist=actor_obs["num_hist"],
            ),
        ),
    )


__all__ = ["build_algorithm_cfg", "build_obs_cfg", "build_policy_cfg"]
