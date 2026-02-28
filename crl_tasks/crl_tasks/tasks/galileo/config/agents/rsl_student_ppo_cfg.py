"""Student PPO runner configuration for Galileo CRL tasks."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..defaults import GalileoDefaults
from .rsl_rl_cfg import (
    CRLRslRlActorCfg,
    CRLRslRlOnPolicyRunnerCfg,
    CRLRslRlPpoActorCriticCfg,
    CRLRslRlPpoAlgorithmCfg,
    CRLRslRlStateHistEncoderCfg,
)


def _algo_cfg_for(role: str) -> CRLRslRlPpoAlgorithmCfg:
    """Build algorithm config from GalileoDefaults.

    Args:
        role: "teacher" | "student"
    """
    algo_key = getattr(GalileoDefaults.algorithm, "name", "fppo")
    class_name = GalileoDefaults.algorithm.class_name_map.get(algo_key, algo_key)
    params: dict = {}
    params.update(getattr(GalileoDefaults.algorithm, "base", {}))
    params.update(getattr(GalileoDefaults.algorithm, "per_algo", {}).get(algo_key, {}))
    if role == "teacher":
        params.update(getattr(GalileoDefaults.algorithm, "teacher_override", {}))
    elif role == "student":
        params.update(getattr(GalileoDefaults.algorithm, "student_override", {}))
    return CRLRslRlPpoAlgorithmCfg(class_name=class_name, **params)


def _obs_cfg_for(role: str) -> tuple[dict, dict]:
    """Build actor/critic observation layout from GalileoDefaults."""
    obs_cfg = getattr(GalileoDefaults.obs, role)
    actor_cfg = dict(
        num_prop=obs_cfg.actor_num_prop,
        num_scan=obs_cfg.actor_num_scan,
        num_priv_explicit=obs_cfg.actor_num_priv_explicit,
        num_priv_latent=obs_cfg.actor_num_priv_latent,
        num_hist=obs_cfg.actor_num_hist,
    )
    critic_cfg = dict(
        critic_num_prop=obs_cfg.critic_num_prop,
        critic_num_scan=obs_cfg.critic_num_scan,
        critic_num_priv_explicit=obs_cfg.critic_num_priv_explicit,
        critic_num_priv_latent=obs_cfg.critic_num_priv_latent,
        critic_num_hist=obs_cfg.critic_num_hist,
    )
    return actor_cfg, critic_cfg


_STUDENT_ACTOR_OBS, _STUDENT_CRITIC_OBS = _obs_cfg_for("student")


@configclass
class GalileoCRLStudentPPORunnerCfg(CRLRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "galileo_fppo"
    empirical_normalization = False
    policy = CRLRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        num_prop=_STUDENT_ACTOR_OBS["num_prop"],
        num_scan=_STUDENT_ACTOR_OBS["num_scan"],
        num_priv_explicit=_STUDENT_ACTOR_OBS["num_priv_explicit"],
        num_priv_latent=_STUDENT_ACTOR_OBS["num_priv_latent"],
        num_hist=_STUDENT_ACTOR_OBS["num_hist"],
        encode_scan_for_critic=(_STUDENT_CRITIC_OBS["critic_num_scan"] > 0),
        critic_num_prop=_STUDENT_CRITIC_OBS["critic_num_prop"],
        critic_num_scan=_STUDENT_CRITIC_OBS["critic_num_scan"],
        critic_num_priv_explicit=_STUDENT_CRITIC_OBS["critic_num_priv_explicit"],
        critic_num_priv_latent=_STUDENT_CRITIC_OBS["critic_num_priv_latent"],
        critic_num_hist=_STUDENT_CRITIC_OBS["critic_num_hist"],
        critic_scan_encoder_dims=[128, 64, 32]
        if _STUDENT_CRITIC_OBS["critic_num_scan"] > 0
        else None,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[],
        activation="elu",
        actor=CRLRslRlActorCfg(
            class_name="Actor",
            num_prop=_STUDENT_ACTOR_OBS["num_prop"],
            num_scan=_STUDENT_ACTOR_OBS["num_scan"],
            num_priv_explicit=_STUDENT_ACTOR_OBS["num_priv_explicit"],
            num_priv_latent=_STUDENT_ACTOR_OBS["num_priv_latent"],
            num_hist=_STUDENT_ACTOR_OBS["num_hist"],
            state_history_encoder=CRLRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
                num_prop=_STUDENT_ACTOR_OBS["num_prop"],
                num_hist=_STUDENT_ACTOR_OBS["num_hist"],
            ),
        ),
    )
    algorithm = _algo_cfg_for("student")


__all__ = ["GalileoCRLStudentPPORunnerCfg"]
