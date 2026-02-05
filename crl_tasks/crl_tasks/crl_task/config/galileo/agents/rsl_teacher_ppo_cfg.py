from crl_tasks.crl_task.config.galileo.agents.crl_rl_cfg import (
CRLRslRlOnPolicyRunnerCfg,
CRLRslRlPpoActorCriticCfg,
CRLRslRlActorCfg,
CRLRslRlStateHistEncoderCfg,
CRLRslRlPpoAlgorithmCfg
)
from isaaclab.utils import configclass
from crl_tasks.crl_task.config.galileo.config_summary import ConfigSummary


def _algo_cfg_for(role: str) -> CRLRslRlPpoAlgorithmCfg:
    """Build algorithm cfg from ConfigSummary.

    role: "teacher" | "student"
    """
    algo_key = getattr(ConfigSummary.algorithm, "name", "fppo")
    class_name = ConfigSummary.algorithm.class_name_map.get(algo_key, algo_key)
    params = {}
    params.update(getattr(ConfigSummary.algorithm, "base", {}))
    params.update(getattr(ConfigSummary.algorithm, "per_algo", {}).get(algo_key, {}))
    if role == "teacher":
        params.update(getattr(ConfigSummary.algorithm, "teacher_override", {}))
    elif role == "student":
        params.update(getattr(ConfigSummary.algorithm, "student_override", {}))
    return CRLRslRlPpoAlgorithmCfg(class_name=class_name, **params)

@configclass
class GalileoCRLTeacherPPORunnerCfg(CRLRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "galileo_fppo"
    empirical_normalization = False
    policy = CRLRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        num_prop=48,
        num_scan=132,
        num_priv_explicit=5,
        encode_scan_for_critic=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
        priv_encoder_dims = [],
        activation="elu",
        actor = CRLRslRlActorCfg(
            class_name = "Actor",
            num_prop=48,
            num_scan=132,
            num_priv_explicit=5,
            state_history_encoder = CRLRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder",
                num_prop=48,
            )
        ),
    )
    algorithm = _algo_cfg_for("teacher")
