# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.rsl_rl.modules.actor_critic_with_encoder import ActorCriticRMA as ActorCritic
from scripts.rsl_rl.storage.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        cost_gamma=None,
        cost_lam=None,
        value_loss_coef=1.0,
        cost_value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Cost advantage normalization
        normalize_cost_advantage: bool = False,
        # VAE auxiliary parameters
        learning_rate_vae: float = 2.0e-3,
        vae_beta: float = 0.01,
        vae_beta_min: float = 1.0e-4,
        vae_beta_max: float = 0.1,
        vae_desired_recon_loss: float = 0.1,
        derived_action_loss_weight: float = 0.0,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Optional VAE optimizer (auxiliary)
        self.vae_optimizer = None
        if getattr(self.policy, "vae_enabled", False) and getattr(self.policy, "vae", None) is not None:
            vae_lr = getattr(self.policy, "vae_learning_rate", None)
            if vae_lr is None:
                vae_lr = learning_rate
            self.vae_optimizer = optim.Adam(self.policy.vae.parameters(), lr=vae_lr)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = gamma if cost_gamma is None else cost_gamma
        self.cost_lam = lam if cost_lam is None else cost_lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.normalize_cost_advantage = normalize_cost_advantage
        # VAE parameters
        self.learning_rate_vae = learning_rate_vae
        self.vae_beta = vae_beta
        self.vae_beta_min = vae_beta_min
        self.vae_beta_max = vae_beta_max
        self.vae_desired_recon_loss = vae_desired_recon_loss
        self.derived_action_loss_weight = derived_action_loss_weight
        # VAE slices
        self._critic_obs_slices = {}
        self._amp_obs_slices = {}

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        vae_actor_obs_shape=None,
        vae_obs_history_shape=None,
        next_actor_obs_shape=None,
        amp_obs_shape=None,
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            self.device,
            actor_obs_shape=vae_actor_obs_shape,
            vae_obs_history_shape=vae_obs_history_shape,
            next_actor_obs_shape=next_actor_obs_shape,
            amp_obs_shape=amp_obs_shape,
        )

    def set_vae_obs_slices(self, critic_obs_slices: dict[str, slice] | None, amp_obs_slices: dict[str, slice] | None):
        self._critic_obs_slices = critic_obs_slices or {}
        self._amp_obs_slices = amp_obs_slices or {}

    def _resolve_critic_obs_slice(self, name: str, fallback: slice) -> slice:
        return self._critic_obs_slices.get(name, fallback)

    def _resolve_amp_obs_slice(self, name: str, fallback: slice) -> slice:
        return self._amp_obs_slices.get(name, fallback)

    def act(self, obs, critic_obs, hist_encoding=False, actor_obs=None, vae_obs_history=None):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs, hist_encoding).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.actor_observations = actor_obs
        self.transition.vae_obs_history = vae_obs_history
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, infos, costs=None, next_actor_obs=None, next_amp_obs=None):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        if costs is None:
            costs = torch.zeros_like(rewards)
        self.transition.cost_rewards = costs.clone()
        self.transition.dones = dones
        self.transition.next_actor_observations = next_actor_obs
        self.transition.amp_observations = next_amp_obs

        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.cost_rewards += self.cost_gamma * torch.squeeze(self.transition.cost_values * time_outs, 1)

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            last_cost_values=last_cost_values,
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_lam,
            normalize_cost_advantage=self.normalize_cost_advantage
            and not self.normalize_advantage_per_mini_batch,
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_vae_vel_loss = 0.0
        mean_vae_mass_loss = 0.0
        mean_vae_decode_loss = 0.0
        mean_vae_kl_loss = 0.0
        mean_vae_beta = 0.0
        mean_derived_action_loss = 0.0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            actor_obs_batch,
            vae_obs_history_batch,
            next_actor_obs_batch,
            amp_obs_batch,
        ) in generator:

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                    if self.normalize_cost_advantage:
                        cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (
                            cost_advantages_batch.std() + 1e-8
                        )

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[2]
            )
            # -- entropy
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

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

            # Cost value function loss
            if self.use_clipped_value_loss:
                cost_value_clipped = cost_values_batch + (cost_value_batch - cost_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Optional VAE update
            if (
                self.vae_optimizer is not None
                and getattr(self.policy, "vae_enabled", False)
                and self.policy.vae is not None
                and vae_obs_history_batch is not None
                and next_actor_obs_batch is not None
            ):
                (
                    _vae_code,
                    vae_code_vel,
                    vae_code_mass,
                    _vae_code_latent,
                    vae_decoded,
                    _vae_mean_vel,
                    _vae_logvar_vel,
                    vae_mean_latent,
                    vae_logvar_latent,
                    _vae_mean_mass,
                    _vae_logvar_mass,
                ) = self.policy.vae.cenet_forward(vae_obs_history_batch, deterministic=False)

                vel_slice = self._resolve_critic_obs_slice("base_lin_vel", slice(0, 3))
                mass_slice = self._resolve_critic_obs_slice("random_mass", slice(-1, None))
                vae_vel_target = critic_obs_batch[:, vel_slice]
                vae_mass_target = critic_obs_batch[:, mass_slice]
                vae_decode_target = next_actor_obs_batch
                recon_dim = min(vae_decoded.shape[1], vae_decode_target.shape[1])
                vae_decoded = vae_decoded[:, :recon_dim]
                vae_decode_target = vae_decode_target[:, :recon_dim]

                loss_recon_vel = nn.MSELoss()(vae_code_vel, vae_vel_target)
                loss_recon_mass = nn.MSELoss()(vae_code_mass, vae_mass_target)
                loss_recon_decode = nn.MSELoss()(vae_decoded, vae_decode_target)
                loss_recon = loss_recon_vel + loss_recon_mass + loss_recon_decode

                k = math.exp(self.learning_rate_vae * (self.vae_desired_recon_loss - loss_recon_decode.item()))
                self.vae_beta = max(self.vae_beta_min, min(self.vae_beta_max, k * self.vae_beta))

                kl_div = -0.5 * torch.sum(
                    1 + vae_logvar_latent - vae_mean_latent.pow(2) - vae_logvar_latent.exp(), dim=1
                )
                kl_loss = self.vae_beta * torch.mean(kl_div)

                derived_action_loss = torch.tensor(0.0, device=self.device)
                if (
                    self.derived_action_loss_weight > 0.0
                    and actor_obs_batch is not None
                    and amp_obs_batch is not None
                    and hasattr(self.policy, "get_derived_action")
                ):
                    derived = self.policy.get_derived_action(actor_obs_batch)
                    if derived is not None:
                        mu, sigma, _ = derived
                        foot_slice = self._resolve_amp_obs_slice("foot_positions", slice(27, 39))
                        foot_pos = amp_obs_batch[:, foot_slice].view(mu.shape[0], mu.shape[1], 3)
                        target_xy = foot_pos[..., :2]
                        sigma = torch.clamp(sigma, min=1.0e-4)
                        sigma_sq = sigma**2
                        diff_sq = torch.sum((target_xy - mu).pow(2), dim=-1)
                        derived_action_loss = (diff_sq / (2.0 * sigma_sq) + torch.log(sigma_sq)).mean()

                vae_loss = loss_recon + kl_loss + self.derived_action_loss_weight * derived_action_loss

                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                self.vae_optimizer.step()

                mean_vae_vel_loss += loss_recon_vel.item()
                mean_vae_mass_loss += loss_recon_mass.item()
                mean_vae_decode_loss += loss_recon_decode.item()
                mean_vae_kl_loss += kl_loss.item()
                mean_vae_beta += float(self.vae_beta)
                mean_derived_action_loss += derived_action_loss.item()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if num_updates > 0 and self.vae_optimizer is not None:
            mean_vae_vel_loss /= num_updates
            mean_vae_mass_loss /= num_updates
            mean_vae_decode_loss /= num_updates
            mean_vae_kl_loss /= num_updates
            mean_vae_beta /= num_updates
            mean_derived_action_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.vae_optimizer is not None:
            loss_dict["vae_vel_loss"] = mean_vae_vel_loss
            loss_dict["vae_mass_loss"] = mean_vae_mass_loss
            loss_dict["vae_decode_loss"] = mean_vae_decode_loss
            loss_dict["vae_kl_loss"] = mean_vae_kl_loss
            loss_dict["vae_beta"] = mean_vae_beta
            loss_dict["derived_action_loss"] = mean_derived_action_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
