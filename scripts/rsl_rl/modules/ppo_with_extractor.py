
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
import math

from .actor_critic_with_encoder import ActorCriticRMA
from rsl_rl.algorithms import PPO

class PPOWithExtractor(PPO):
    """带特权状态估计器与历史对齐正则的 PPO。"""
    policy: ActorCriticRMA

    def __init__(
        self,
        policy,
        estimator,
        estimator_paras,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        priv_reg_coef_schedual = [0, 0, 0],
        multi_gpu_cfg: dict | None = None,
        # MoE/gating parameters
        moe_aux_scales: dict | None = None,
        moe_min_usage: float = 0.15,
        gating_temperature_schedule: list[float] | tuple[float, float, float] | None = None,
    ):
        super().__init__(
            policy, 
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
            normalize_advantage_per_mini_batch,
            # RND parameters
            rnd_cfg,
            # Symmetry parameters
            symmetry_cfg,
            # Distributed training parameters
            multi_gpu_cfg,
            )

        self.estimator: nn.Module = estimator
        print(f"estimator MLP: {estimator}")

        self.priv_states_dim = estimator_paras["num_priv_explicit"]
        # The first num_priv_hurdles dims of priv_explicit are hurdle semantics for gating; never estimated.
        num_priv_hurdles = int(estimator_paras.get("num_priv_hurdles", 0))
        if num_priv_hurdles <= 0 and hasattr(self.policy.actor, "gating_input_indices"):
            num_priv_hurdles = len(getattr(self.policy.actor, "gating_input_indices"))
        self.num_priv_hurdles = num_priv_hurdles
        self.priv_states_dim_other = max(int(self.priv_states_dim) - int(self.num_priv_hurdles), 0)
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        self.hist_encoder_optimizer = optim.Adam(self.policy.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0
        # MoE/gating configs
        self.moe_aux_scales = moe_aux_scales or {}
        self.moe_aux_enabled = any(v != 0 for v in self.moe_aux_scales.values())
        self.moe_min_usage = moe_min_usage
        if gating_temperature_schedule is not None and len(gating_temperature_schedule) >= 3:
            start_t, end_t, steps = gating_temperature_schedule[:3]
            self.gating_temp_schedule = (float(start_t), float(end_t), max(float(steps), 1.0))
            if hasattr(self.policy.actor, "gating_temperature"):
                self.policy.actor.gating_temperature = float(start_t)
        else:
            self.gating_temp_schedule = None
        self.last_gating_temperature = getattr(self.policy.actor, "gating_temperature", None)


    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, privileged_obs=None, actions_shape=None):
        """Accept legacy signature and create storage compatible with rsl-rl>=3.1."""
        if actions_shape is None:
            actions_shape = (self.policy.num_actions,) if hasattr(self.policy, "num_actions") else (1,)
        if isinstance(actions_shape, list):
            actions_shape = tuple(actions_shape)
        if isinstance(obs, TensorDict):
            obs_td = obs.to(self.device)
        else:
            # obs is shape list like [num_obs]
            obs_shape = obs if isinstance(obs, (list, tuple)) else (obs,)
            obs_tensor = torch.zeros((num_envs, *obs_shape), device=self.device)
            obs_dict = {"obs": obs_tensor}
            if privileged_obs is not None and privileged_obs[0] > 0:
                priv_tensor = torch.zeros((num_envs, *privileged_obs), device=self.device)
                obs_dict["critic_obs"] = priv_tensor
            obs_td = TensorDict(obs_dict, batch_size=[num_envs], device=self.device)
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs_td, actions_shape)

    def act(self, obs, critic_obs, hist_encoding=False):
        """采样动作并缓存过渡，用估计器可选替换特权显式段。"""
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        obs_td = TensorDict({"obs": obs}, batch_size=[obs.shape[0]], device=self.device)
        if critic_obs is not None:
            obs_td["critic_obs"] = critic_obs
        # compute the actions and values
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            start = self.num_prop + self.num_scan + self.num_priv_hurdles
            end = self.num_prop + self.num_scan + self.priv_states_dim
            if self.priv_states_dim_other > 0:
                obs_est[:, start:end] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.policy.act(obs, hist_encoding).detach()

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs_td
        self.transition.privileged_observations = critic_obs

        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        """收集环境一步数据，计算 RND、处理超时。"""
        obs_td = TensorDict({"obs": obs}, batch_size=[obs.shape[0]], device=self.device)
        # Update the normalizers
        if hasattr(self.policy, "update_normalization"):
            self.policy.update_normalization(obs_td)
        if self.rnd:
            self.rnd.update_normalization(obs_td)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs_td)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, critic_obs):
        last_values = self.policy.evaluate(critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def broadcast_parameters(self):
        """同步 policy/估计器/RND 参数到所有 GPU。"""
        if not self.is_multi_gpu:
            return
        state_to_sync = {
            "policy": self.policy.state_dict(),
            "estimator": self.estimator.state_dict(),
        }
        if self.rnd:
            state_to_sync["rnd"] = self.rnd.state_dict()
        obj_list = [state_to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced_state = obj_list[0]
        self.policy.load_state_dict(synced_state["policy"])
        self.estimator.load_state_dict(synced_state["estimator"])
        if self.rnd and "rnd" in synced_state:
            self.rnd.load_state_dict(synced_state["rnd"])

    def reduce_parameters(self):
        """跨 GPU 平均梯度。"""
        if not self.is_multi_gpu:
            return

        modules = [self.policy, self.estimator]
        if self.rnd:
            modules.append(self.rnd)

        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                    param.grad /= self.gpu_world_size
    

    def update(self):  # noqa: C901
        """PPO 主更新：含特权正则、状态估计、可选对称/RND。"""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_priv_reg_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0
        mean_moe_aux_loss = 0
        mean_gate_entropy = 0
        mean_gate_min = 0
        gate_batches = 0
        gate_load_sum = None
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        current_gating_temp = self._update_gating_temperature()

        # iterate over batches (handle both legacy and current rollout generators)
        for batch in generator:
            if len(batch) == 10:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                ) = batch
                rnd_state_batch = None
                # unpack tensordict if present
                if isinstance(obs_batch, TensorDict):
                    critic_obs_batch = obs_batch.get("critic_obs", obs_batch.get("obs"))
                    obs_batch = obs_batch.get("obs")
                else:
                    critic_obs_batch = obs_batch
            else:
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                    rnd_state_batch,
                ) = batch

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # zero grads for all optimizers before computing losses
            self.optimizer.zero_grad()
            self.estimator_optimizer.zero_grad()
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

            # Estimator
            priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # batch 中原始 priv_states 仍保留
            start = self.num_prop + self.num_scan + self.num_priv_hurdles
            end = self.num_prop + self.num_scan + self.priv_states_dim
            if self.priv_states_dim_other > 0:
                estimator_loss = (priv_states_predicted - obs_batch[:, start:end]).pow(2).mean()
            else:
                estimator_loss = torch.zeros((), device=self.device)
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + \
                self.value_loss_coef * value_loss -\
                self.entropy_coef * entropy_batch.mean() + \
                priv_reg_coef * priv_reg_loss

            moe_aux_loss = None
            if self.moe_aux_enabled:
                gate_probs = self._compute_gate_probs(obs_batch)
                if gate_probs is not None:
                    loads = gate_probs.mean(dim=0)
                    if gate_load_sum is None:
                        gate_load_sum = loads.detach()
                    else:
                        gate_load_sum = gate_load_sum + loads.detach()
                    target = 1.0 / gate_probs.shape[1]
                    load_balance_loss = ((loads - target) ** 2).sum()
                    entropy_loss = (-(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1)).mean()
                    diversity_loss = torch.relu(self.moe_min_usage - loads.min())
                    moe_aux_loss = (
                        self.moe_aux_scales.get("balance", 0.0) * load_balance_loss
                        + self.moe_aux_scales.get("entropy", 0.0) * entropy_loss
                        + self.moe_aux_scales.get("diversity", 0.0) * diversity_loss
                    )
                    loss = loss + moe_aux_loss
                    mean_gate_entropy += entropy_loss.detach().item()
                    mean_gate_min += loads.min().detach().item()
                    gate_batches += 1

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                if rnd_state_batch is None:
                    rnd_state_batch = obs_batch
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)


            estimator_loss.backward()
            loss.backward()

            if self.rnd:
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            if self.rnd_optimizer:
                nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), self.max_grad_norm)  # type: ignore

            self.estimator_optimizer.step()
            self.optimizer.step()

            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.mean().item()
            mean_estimator_loss += estimator_loss.item()
            if moe_aux_loss is not None:
                mean_moe_aux_loss += moe_aux_loss.item()

            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        if self.moe_aux_enabled and gate_batches > 0:
            mean_moe_aux_loss /= gate_batches
            mean_gate_entropy /= gate_batches
            mean_gate_min /= gate_batches
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        self.update_counter()
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "priv_reg": mean_priv_reg_loss,
            "entropy": mean_entropy,
            'estimator':mean_estimator_loss,
            'priv_reg_coef': priv_reg_coef
        }
        if self.moe_aux_enabled and gate_batches > 0:
            loss_dict["moe_aux"] = mean_moe_aux_loss
            loss_dict["gate_entropy"] = mean_gate_entropy
            loss_dict["gate_min_usage"] = mean_gate_min
            if gate_load_sum is not None:
                mean_loads = gate_load_sum / gate_batches
                expert_names = getattr(self.policy.actor, "expert_names", None)
                for i in range(mean_loads.numel()):
                    name = (
                        expert_names[i]
                        if expert_names is not None and i < len(expert_names)
                        else f"expert_{i}"
                    )
                    loss_dict[f"gate_usage/{name}"] = mean_loads[i].item()
            if current_gating_temp is not None:
                loss_dict["gating_temperature"] = current_gating_temp
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def update_counter(self):
        self.counter += 1

    def update_dagger(self):
        """DAgger 式自蒸馏：强制历史编码对齐特权隐式编码。"""
        mean_hist_latent_loss = 0
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if len(batch) == 10:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                ) = batch
                if isinstance(obs_batch, TensorDict):
                    obs_batch = obs_batch.get("obs")
                critic_obs_batch = obs_batch
            else:
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hid_states_batch,
                    masks_batch,
                    rnd_state_batch,
                ) = batch
                if isinstance(obs_batch, TensorDict):
                    critic_obs_batch = obs_batch.get("critic_obs", obs_batch.get("obs"))
                    obs_batch = obs_batch.get("obs")
            with torch.inference_mode():
                self.policy.act(obs_batch, 
                                hist_encoding=True, 
                                masks=masks_batch, 
                                hidden_states=hid_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def _compute_gate_probs(self, obs_batch: torch.Tensor) -> torch.Tensor | None:
        """使用当前 batch 的特权显式段计算 gating 分布，支持反向传播。"""
        actor = getattr(self.policy, "actor", None)
        if actor is None or not hasattr(actor, "gating_input_indices"):
            return None
        start = actor.num_prop + actor.num_scan
        end = start + actor.num_priv_explicit
        obs_priv_explicit = obs_batch[:, start:end]
        gate_feat = obs_priv_explicit[:, actor.gating_input_indices]
        return actor._compute_gate(gate_feat)

    def _update_gating_temperature(self) -> float | None:
        """余弦退火 gating 温度：早期高温探索，后期低温聚焦。"""
        if self.gating_temp_schedule is None or not hasattr(self.policy.actor, "gating_temperature"):
            return getattr(self.policy.actor, "gating_temperature", None)
        start_t, end_t, steps = self.gating_temp_schedule
        progress = min(max(self.counter / steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_temp = end_t + (start_t - end_t) * cosine
        self.policy.actor.gating_temperature = current_temp
        self.last_gating_temperature = current_temp
        return current_temp
