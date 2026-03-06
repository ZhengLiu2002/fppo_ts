from __future__ import annotations

import os
import time
import torch
import inspect
import statistics
from collections import deque

import rsl_rl
from rsl_rl.env import VecEnv

try:
    from rsl_rl.modules import EmpiricalNormalization  # type: ignore
except ImportError:
    import torch

    class EmpiricalNormalization(torch.nn.Module):
        """Fallback normalizer when rsl_rl version lacks EmpiricalNormalization."""

        def __init__(self, shape, until=1.0e8):
            super().__init__()
            self.register_buffer("count", torch.tensor(0.0))
            self.register_buffer("mean", torch.zeros(shape))
            self.register_buffer("var", torch.ones(shape))
            self.until = until

        def forward(self, x):
            if self.training and self.count < self.until:
                # Simple running mean/var update
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                batch_count = x.shape[0]

                delta = batch_mean - self.mean
                tot_count = self.count + batch_count

                new_mean = self.mean + delta * batch_count / tot_count
                m_a = self.var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
                new_var = M2 / tot_count

                self.mean = new_mean
                self.var = new_var
                self.count = tot_count

            std = torch.sqrt(self.var + 1e-8)
            return (x - self.mean) / std


from .actor_critic_with_encoder import ActorCriticRMA
from rsl_rl.utils import store_code_state
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from scripts.rsl_rl.algorithms.registry import (
    get_algorithm_class,
    get_algorithm_spec,
    strict_algorithm_cfg_enabled,
    validate_algorithm_cfg,
)
from scripts.rsl_rl.constraint_utils import ConstraintNormalizer

_RUNNER_ONLY_ALG_KEYS = {
    "class_name",
    "dagger_update_freq",
    "rnd_cfg",
    "symmetry_cfg",
    "constraint_normalization",
    "constraint_norm_beta",
    "constraint_norm_min_scale",
    "constraint_norm_max_scale",
    "constraint_norm_clip",
    "constraint_proxy_delta",
    "constraint_agg_tau",
    "constraint_scale_by_gamma",
    "constraint_cost_scale",
}

_KEY_CURVE_MIRROR_MAP = {
    "Train/mean_reward": "Compare/Core/reward",
    "Train/mean_episode_length": "Compare/Core/episode_length",
    "Metrics/base_velocity/error_vel_xy": "Compare/Tracking/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw": "Compare/Tracking/error_vel_yaw",
    "Episode_Reward/track_lin_vel_xy_exp": "Compare/Tracking/track_lin_vel_xy",
    "Episode_Reward/track_ang_vel_z_exp": "Compare/Tracking/track_ang_vel_z",
    "Cost/mean_cost_return": "Compare/Cost/mean_cost_return",
    "Cost/cost_limit_margin": "Compare/Cost/cost_limit_margin",
    "Cost/cost_violation_rate": "Compare/Cost/cost_violation_rate",
    "Episode_Cost/prob_gait_pattern": "Compare/Gait/prob_gait_pattern",
    "Episode_Cost/symmetric": "Compare/Gait/symmetric",
    "Episode_Cost/contact_velocity": "Compare/Gait/contact_velocity",
    "Episode_Cost/prob_com_frame": "Compare/Stability/prob_com_frame",
    "Episode_Cost/prob_joint_pos": "Compare/Cost/prob_joint_pos",
    "Episode_Cost/prob_joint_vel": "Compare/Cost/prob_joint_vel",
    "Episode_Cost/prob_joint_torque": "Compare/Cost/prob_joint_torque",
}


class OnPolicyRunnerWithExtractor(OnPolicyRunner):
    """纯算法训练 Runner，支持 PPO/FPPO/CPO/Distillation 等流程。"""

    @staticmethod
    def _expected_obs_dim(policy_cfg: dict) -> int | None:
        num_prop = policy_cfg.get("num_prop", None)
        if num_prop is None:
            return None
        try:
            num_prop = int(num_prop)
        except Exception:
            return None
        num_scan = int(policy_cfg.get("num_scan", 0) or 0)
        num_priv_explicit = int(policy_cfg.get("num_priv_explicit", 0) or 0)
        num_priv_latent = int(policy_cfg.get("num_priv_latent", 0) or 0)
        num_hist = int(policy_cfg.get("num_hist", 0) or 0)
        return num_prop + num_scan + num_priv_explicit + num_priv_latent + num_prop * num_hist

    @staticmethod
    def _infer_num_cost_terms(env: VecEnv) -> int | None:
        cost_manager = getattr(env.unwrapped, "cost_manager", None)
        if cost_manager is None:
            return None
        term_names = getattr(cost_manager, "active_terms", None)
        if isinstance(term_names, (list, tuple)) and len(term_names) > 0:
            return int(len(term_names))
        return None

    @staticmethod
    def _infer_cost_term_names(env: VecEnv) -> list[str] | None:
        cost_manager = getattr(env.unwrapped, "cost_manager", None)
        if cost_manager is None:
            return None
        term_names = getattr(cost_manager, "active_terms", None)
        if isinstance(term_names, (list, tuple)) and len(term_names) > 0:
            return sorted(str(name) for name in term_names)
        return None

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.alg_cfg_full = dict(self.alg_cfg)
        self.estimator_cfg = train_cfg.get("estimator")
        self.depth_encoder_cfg = train_cfg.get("depth_encoder")
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.mean_hist_latent_loss = 0.0
        self._key_curve_mirror_map = dict(_KEY_CURVE_MIRROR_MAP)
        self._configure_multi_gpu()
        self._constraint_normalizer = None
        self._constraint_scale_by_gamma = bool(
            self.alg_cfg_full.get("constraint_scale_by_gamma", False)
        )
        self._constraint_scale = self.alg_cfg_full.get("constraint_cost_scale", None)
        self._constraint_gamma = self.alg_cfg_full.get("cost_gamma")
        if self._constraint_gamma is None:
            self._constraint_gamma = self.alg_cfg_full.get("gamma")
        if self.alg_cfg_full.get("constraint_normalization", False):
            self._constraint_normalizer = ConstraintNormalizer.from_cfg(
                self.alg_cfg_full, device=self.device
            )
        self._constraint_term_names = self._infer_cost_term_names(self.env)
        named_constraint_limits = self.alg_cfg_full.get("constraint_limits", None)
        if isinstance(named_constraint_limits, dict):
            # Support named per-constraint limits to avoid fragile order coupling.
            default_limit = float(self.alg_cfg_full.get("cost_limit", 0.0))
            if self._constraint_term_names is None:
                ordered_names = sorted(str(name) for name in named_constraint_limits.keys())
            else:
                ordered_names = self._constraint_term_names
            ordered_limits = [
                float(named_constraint_limits.get(name, default_limit)) for name in ordered_names
            ]
            self.alg_cfg["constraint_limits"] = ordered_limits
            self.alg_cfg_full["constraint_limits"] = ordered_limits

        alg_class_name = self.alg_cfg["class_name"]
        alg_spec = get_algorithm_spec(alg_class_name)
        self.training_type = alg_spec.training_type

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic 强化学习模式（例如 PPO）
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # 策略蒸馏：教师观测
            else:
                self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        # Multi-constraint fairness baseline: for RL algorithms, set the number
        # of cost-value heads to the number of active cost terms unless user
        # explicitly overrides it.
        if self.training_type == "rl":
            inferred_heads = self._infer_num_cost_terms(self.env)
            if inferred_heads is None:
                limits = self.alg_cfg_full.get("constraint_limits", None)
                if isinstance(limits, (list, tuple)) and len(limits) > 0:
                    inferred_heads = len(limits)
            if inferred_heads is not None and inferred_heads > 0:
                configured_heads = self.policy_cfg.get("num_cost_heads", None)
                if configured_heads is None or int(configured_heads) <= 1:
                    self.policy_cfg["num_cost_heads"] = int(inferred_heads)
        expected_obs = self._expected_obs_dim(self.policy_cfg)
        if expected_obs is not None and num_obs != expected_obs:
            raise ValueError(
                f"Policy obs dim mismatch: env provides {num_obs}, "
                f"but policy_cfg expects {expected_obs} "
                f"(num_prop={self.policy_cfg.get('num_prop')}, "
                f"num_scan={self.policy_cfg.get('num_scan', 0)}, "
                f"num_priv_explicit={self.policy_cfg.get('num_priv_explicit', 0)}, "
                f"num_priv_latent={self.policy_cfg.get('num_priv_latent', 0)}, "
                f"num_hist={self.policy_cfg.get('num_hist', 0)})."
            )
        expected_critic = self._expected_obs_dim(
            {
                "num_prop": self.policy_cfg.get("critic_num_prop"),
                "num_scan": self.policy_cfg.get("critic_num_scan", 0),
                "num_priv_explicit": self.policy_cfg.get("critic_num_priv_explicit", 0),
                "num_priv_latent": self.policy_cfg.get("critic_num_priv_latent", 0),
                "num_hist": self.policy_cfg.get("critic_num_hist", 0),
            }
        )
        if (
            expected_critic is not None
            and self.privileged_obs_type is not None
            and num_privileged_obs != expected_critic
        ):
            raise ValueError(
                f"Critic obs dim mismatch: env provides {num_privileged_obs}, "
                f"but policy_cfg expects {expected_critic} "
                f"(critic_num_prop={self.policy_cfg.get('critic_num_prop')}, "
                f"critic_num_scan={self.policy_cfg.get('critic_num_scan', 0)}, "
                f"critic_num_priv_explicit={self.policy_cfg.get('critic_num_priv_explicit', 0)}, "
                f"critic_num_priv_latent={self.policy_cfg.get('critic_num_priv_latent', 0)}, "
                f"critic_num_hist={self.policy_cfg.get('critic_num_hist', 0)})."
            )
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticRMA = policy_class(
            num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError(
                    "Observations for the key 'rnd_state' not found in infos['observations']."
                )
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        def _filter_kwargs(klass, cfg_dict: dict) -> dict:
            sig = inspect.signature(klass.__init__)
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
            return {k: v for k, v in cfg_dict.items() if k in allowed}

        if self.estimator_cfg is not None or self.depth_encoder_cfg is not None:
            raise ValueError(
                "Estimator/depth encoder configs are set, but this runner uses pure algorithms only."
            )
        self.learn = self.learn_rl

        # 纯算法训练
        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        self.dagger_update_freq = alg_cfg.pop("dagger_update_freq", 1)
        alg_class = get_algorithm_class(alg_class_name)
        validate_algorithm_cfg(
            alg_class,
            self.alg_cfg_full,
            extra_allowed_keys=_RUNNER_ONLY_ALG_KEYS | set(alg_spec.extra_cfg_keys),
            strict=strict_algorithm_cfg_enabled(),
        )
        alg_kwargs = _filter_kwargs(alg_class, alg_cfg)
        if "device" not in alg_kwargs:
            alg_kwargs["device"] = self.device
        if (
            "multi_gpu_cfg" in inspect.signature(alg_class.__init__).parameters
            and "multi_gpu_cfg" not in alg_kwargs
        ):
            alg_kwargs["multi_gpu_cfg"] = self.multi_gpu_cfg
        self.alg = alg_class(policy=policy, **alg_kwargs)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(
                self.device
            )
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            (num_obs,),
            (num_privileged_obs,),
            (self.env.num_actions,),
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn_rl(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ):  # noqa: C901
        """标准 RL 训练循环：收集 rollout -> 更新 PPO -> 记录日志。"""
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError(
                    "Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'."
                )

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if getattr(self.alg, "rnd", None):
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            if hasattr(self.alg, "broadcast_parameters"):
                self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs, hist_encoding)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # sanitize rewards to avoid NaN/Inf propagating into returns/advantages
                    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
                    rewards = torch.clamp(rewards, min=-1.0e3, max=1.0e3)
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    privileged_obs = torch.nan_to_num(
                        privileged_obs, nan=0.0, posinf=1.0e6, neginf=-1.0e6
                    )

                    # process the step
                    costs, cost_terms = self._extract_costs(infos, rewards)
                    if costs is not None:
                        costs = torch.nan_to_num(costs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
                        costs = torch.clamp(costs, min=-1.0e3, max=1.0e3)
                    self.alg.process_env_step(
                        obs, rewards, dones, infos, costs=costs, cost_terms=cost_terms
                    )

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if getattr(self.alg, "rnd", None) else None
                    )

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if getattr(self.alg, "rnd", None):
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if getattr(self.alg, "rnd", None):
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()
            cost_metrics = getattr(self.alg, "train_metrics", None)
            if hist_encoding and hasattr(self.alg, "update_dagger"):
                print("Updating dagger...")
                self.mean_hist_latent_loss = self.alg.update_dagger()
            loss_dict["hist_latent"] = self.mean_hist_latent_loss

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self._serialize_optimizer(self.alg.optimizer),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if getattr(self.alg, "rnd", None):
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )
        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        model_state = loaded_dict["model_state_dict"]
        try:
            resumed_training = self.alg.policy.load_state_dict(model_state)
        except RuntimeError as exc:
            # Backward compatibility: older checkpoints may use single-head cost critics,
            # while current configs can infer multi-head critics from active constraints.
            if "cost_critic" not in str(exc):
                raise
            policy_state = self.alg.policy.state_dict()
            patched = False
            for key, src_tensor in list(model_state.items()):
                if not (key.startswith("cost_critic.") and key.endswith(".weight")):
                    continue
                if key not in policy_state:
                    continue
                tgt_tensor = policy_state[key]
                if src_tensor.ndim != 2 or tgt_tensor.ndim != 2:
                    continue
                if src_tensor.shape[1] != tgt_tensor.shape[1]:
                    continue
                if src_tensor.shape[0] == tgt_tensor.shape[0]:
                    continue

                bias_key = key.replace(".weight", ".bias")
                if bias_key not in model_state or bias_key not in policy_state:
                    continue
                src_bias = model_state[bias_key]
                tgt_bias = policy_state[bias_key]
                if src_bias.ndim != 1 or tgt_bias.ndim != 1:
                    continue

                # 1->N: replicate the single head. Otherwise copy overlapping heads.
                if src_tensor.shape[0] == 1:
                    new_weight = src_tensor.expand(tgt_tensor.shape[0], -1).clone()
                else:
                    new_weight = tgt_tensor.clone()
                    rows = min(src_tensor.shape[0], tgt_tensor.shape[0])
                    new_weight[:rows, :] = src_tensor[:rows, :]
                if src_bias.shape[0] == 1:
                    new_bias = src_bias.expand(tgt_bias.shape[0]).clone()
                else:
                    new_bias = tgt_bias.clone()
                    rows = min(src_bias.shape[0], tgt_bias.shape[0])
                    new_bias[:rows] = src_bias[:rows]

                model_state[key] = new_weight
                model_state[bias_key] = new_bias
                patched = True

            if not patched:
                raise
            print(
                "[WARN] Checkpoint cost_critic head count mismatched current config; "
                "auto-adapted output layer weights for compatibility."
            )
            resumed_training = self.alg.policy.load_state_dict(model_state)
        if getattr(self.alg, "rnd", None):
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["privileged_obs_norm_state_dict"]
                )
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self._load_optimizer(self.alg.optimizer, loaded_dict.get("optimizer_state_dict"))
            # -- RND optimizer if used
            if getattr(self.alg, "rnd", None):
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:  # noqa: C901
        """Custom logger to support cost metrics and robust key aggregation."""
        def _log_scalar(tag: str, value: float, step: int):
            self.writer.add_scalar(tag, value, step)
            mirror_tag = self._key_curve_mirror_map.get(tag)
            if mirror_tag is not None:
                self.writer.add_scalar(mirror_tag, value, step)

        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information (use union of keys to avoid missing cost metrics)
        ep_string = ""
        ep_infos = locs.get("ep_infos", [])
        if ep_infos:
            keys = set()
            for ep_info in ep_infos:
                keys.update(ep_info.keys())
            for key in sorted(keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                scalar = value.item() if torch.is_tensor(value) else float(value)
                fmt = "{:.6f}" if abs(scalar) < 1.0e-3 else "{:.4f}"
                if "/" in key:
                    _log_scalar(key, scalar, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {fmt.format(scalar)}\n"""
                else:
                    _log_scalar("Episode/" + key, scalar, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {fmt.format(scalar)}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))
        cost_metrics = locs.get("cost_metrics")

        # Log losses
        for key, value in locs["loss_dict"].items():
            _log_scalar(f"Loss/{key}", value, locs["it"])
        if isinstance(cost_metrics, dict) and "base_step_size" in cost_metrics:
            _log_scalar("Loss/learning_rate", float(cost_metrics["base_step_size"]), locs["it"])
        else:
            _log_scalar("Loss/learning_rate", float(self.alg.learning_rate), locs["it"])
        if isinstance(cost_metrics, dict) and "step_size" in cost_metrics:
            _log_scalar("Loss/effective_step_size", float(cost_metrics["step_size"]), locs["it"])
        if isinstance(cost_metrics, dict) and "effective_step_ratio" in cost_metrics:
            _log_scalar("Loss/effective_step_ratio", float(cost_metrics["effective_step_ratio"]), locs["it"])

        # Log optional cost metrics
        cost_string = ""
        if isinstance(cost_metrics, dict) and cost_metrics:
            for key, value in cost_metrics.items():
                scalar = value.item() if torch.is_tensor(value) else float(value)
                _log_scalar(f"Cost/{key}", scalar, locs["it"])
                cost_string += f"""{f"Cost/{key}:":>{pad}} {scalar:.4f}\n"""

        # Log noise std
        _log_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        _log_scalar("Perf/total_fps", fps, locs["it"])
        _log_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        _log_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar(
                    "Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"]
                )
                self.writer.add_scalar(
                    "Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"]
                )
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            _log_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            _log_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        # Terminal log
        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{header.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
        if cost_string:
            log_string += cost_string
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += (
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    @staticmethod
    def _serialize_optimizer(optimizer):
        if optimizer is None:
            return None
        if hasattr(optimizer, "state_dict"):
            return optimizer.state_dict()
        if isinstance(optimizer, dict):
            return {
                key: opt.state_dict() if hasattr(opt, "state_dict") else opt
                for key, opt in optimizer.items()
            }
        if isinstance(optimizer, (list, tuple)):
            return [opt.state_dict() if hasattr(opt, "state_dict") else opt for opt in optimizer]
        return optimizer

    @staticmethod
    def _load_optimizer(optimizer, state):
        if optimizer is None or state is None:
            return
        if hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(state)
            return
        if isinstance(optimizer, dict) and isinstance(state, dict):
            for key, opt in optimizer.items():
                opt_state = state.get(key)
                if opt_state is None or not hasattr(opt, "load_state_dict"):
                    continue
                opt.load_state_dict(opt_state)
            return
        if isinstance(optimizer, (list, tuple)) and isinstance(state, (list, tuple)):
            for opt, opt_state in zip(optimizer, state):
                if hasattr(opt, "load_state_dict"):
                    opt.load_state_dict(opt_state)

    def get_inference_policy(self, device=None):
        """返回 actor 推理函数，若开启归一化则自动预处理输入。"""
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x, *args, **kwargs: self.alg.policy.act_inference(  # noqa: E731
                self.obs_normalizer(x), *args, **kwargs
            )
        return policy

    def _extract_costs(self, infos: dict, rewards: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
        cost = infos.get("cost") if isinstance(infos, dict) else None
        if cost is None:
            return torch.zeros_like(rewards), None
        cost_terms_payload = None
        if isinstance(cost, dict):
            if not cost:
                return torch.zeros_like(rewards), None
            raw_terms: dict[str, torch.Tensor] = {}
            for key, value in cost.items():
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=self.device)
                value = value.to(self.device)
                if value.ndim > 1 and value.shape[-1] > 1:
                    value = value.sum(dim=-1)
                elif value.ndim > 1 and value.shape[-1] == 1:
                    value = value.squeeze(-1)
                raw_terms[str(key)] = torch.clamp(value, min=0.0)
            if self._constraint_normalizer is not None:
                term_dict = self._constraint_normalizer.normalize(raw_terms)
            else:
                term_dict = raw_terms
            term_names = sorted(term_dict.keys())
            term_values = torch.stack([term_dict[name] for name in term_names], dim=-1)
            cost = torch.sum(term_values, dim=-1)
            cost_terms_payload = {"values": term_values, "names": term_names}
        else:
            if self._constraint_normalizer is not None:
                cost, _ = self._constraint_normalizer.aggregate({"total": cost})
        if not torch.is_tensor(cost):
            cost = torch.as_tensor(cost, device=self.device)
        cost = torch.clamp(cost.to(self.device), min=0.0)
        cost_scale = self._get_constraint_scale()
        if cost_scale is not None and cost_scale != 1.0:
            cost = cost * cost_scale
            if cost_terms_payload is not None:
                cost_terms_payload["values"] = cost_terms_payload["values"] * cost_scale
        if cost.ndim > 1 and cost.shape[-1] > 1:
            cost = cost.sum(dim=-1)
        if cost.ndim == 1 and rewards.ndim > 1:
            cost = cost.unsqueeze(-1)
        return cost, cost_terms_payload

    def _get_constraint_scale(self) -> float | None:
        if self._constraint_scale is not None:
            return float(self._constraint_scale)
        if not self._constraint_scale_by_gamma:
            return None
        cost_gamma = self._constraint_gamma
        if cost_gamma is None:
            return None
        if not (0.0 < cost_gamma < 1.0):
            return None
        return 1.0 - float(cost_gamma)
