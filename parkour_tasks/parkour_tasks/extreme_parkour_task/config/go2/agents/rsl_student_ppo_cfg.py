from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlDistillationAlgorithmCfg,
ParkourRslRlDepthEncoderCfg
)
from isaaclab.utils import configclass

@configclass
class UnitreeGo2ParkourStudentPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 100000 
    save_interval = 100
    experiment_name = "unitree_go2_parkour"
    empirical_normalization = False
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
        priv_encoder_dims = [64, 20],
        activation="elu",
        actor = ParkourRslRlActorCfg(
            class_name = "GatedMoEActor",
            state_history_encoder = ParkourRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder" 
            )
        ),
        gating_hidden_dims=[64, 64],
        gating_temperature=2.0,
        gating_input_indices=[0,1,2,3,4,5,6,7],
        gating_top_k=2,
        num_experts=3,
        expert_names=["flat", "jump", "crawl"],
    )
    estimator = ParkourRslRlEstimatorCfg(
            hidden_dims = [128, 64]
    )
    depth_encoder = ParkourRslRlDepthEncoderCfg(
        hidden_dims = 512,
        learning_rate= 1e-3,
        num_steps_per_env = 24*5
    )

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=5,
        learning_rate = 1.5e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        gating_temperature_schedule=[2.0, 0.6, 6000.0],
        moe_aux_scales={"balance": 0.03, "entropy": 0.02, "diversity": 0.05},
        moe_min_usage=0.10,
    )
