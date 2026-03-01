# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
        # Positive-part cost violation regularization
        cost_limit: float = 0.0,
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
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
        self.cost_limit = cost_limit
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.k_value = float(k_value)
        self.k_growth = float(k_growth)
        self.k_max = float(k_max)
        self.train_metrics: dict[str, float] = {}

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
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
        )

    def _all_reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        if self.is_multi_gpu:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    @staticmethod
    def _sanitize_tensor(
        tensor: torch.Tensor,
        *,
        nan: float = 0.0,
        posinf: float = 1.0e6,
        neginf: float = -1.0e6,
        clamp: float | None = None,
    ) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        if clamp is not None:
            tensor = torch.clamp(tensor, min=-clamp, max=clamp)
        return tensor

    def _safe_ratio(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        log_ratio = self._sanitize_tensor(log_ratio, nan=0.0, posinf=20.0, neginf=-20.0, clamp=20.0)
        return torch.exp(log_ratio)

    def _safe_kl(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> torch.Tensor:
        sigma_batch = self._sanitize_tensor(sigma_batch, nan=1.0e-6, posinf=1.0, neginf=1.0e-6)
        old_sigma_batch = self._sanitize_tensor(
            old_sigma_batch, nan=1.0e-6, posinf=1.0, neginf=1.0e-6
        )
        sigma_batch = torch.clamp(sigma_batch, min=1.0e-6)
        old_sigma_batch = torch.clamp(old_sigma_batch, min=1.0e-6)
        mu_batch = self._sanitize_tensor(mu_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6)
        old_mu_batch = self._sanitize_tensor(
            old_mu_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            dim=-1,
        )
        return self._sanitize_tensor(kl, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6)

    def _batch_cost_stats(
        self, cost_returns_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost_returns_batch = self._sanitize_tensor(
            cost_returns_batch, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        cost_return_mean = self._all_reduce_mean(cost_returns_batch.mean())
        cost_violation_rate = self._all_reduce_mean(
            (cost_returns_batch > self.cost_limit).float().mean()
        )
        cost_return_mean = self._sanitize_tensor(
            cost_return_mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
        )
        cost_violation_rate = self._sanitize_tensor(
            cost_violation_rate, nan=0.0, posinf=1.0, neginf=0.0, clamp=1.0
        )
        c_hat = cost_return_mean - self.cost_limit
        c_hat = self._sanitize_tensor(c_hat, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6)
        return cost_return_mean, cost_violation_rate, c_hat

    def _positive_cost_penalty(
        self, cost_surrogate: torch.Tensor, c_hat: torch.Tensor, detach_violation: bool = True
    ) -> torch.Tensor:
        if self.cost_viol_loss_coef <= 0.0 or self.k_value <= 0.0:
            return torch.zeros((), device=cost_surrogate.device, dtype=cost_surrogate.dtype)
        violation = c_hat.detach() if detach_violation else c_hat
        return self.cost_viol_loss_coef * self.k_value * torch.relu(cost_surrogate + violation)

    def _step_constraint_scale(self):
        self.k_value = min(self.k_max, self.k_value * self.k_growth)

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs, hist_encoding).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, infos, costs=None):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        if costs is None:
            costs = torch.zeros_like(rewards)
        self.transition.cost_rewards = costs.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs, 1
            )
            self.transition.cost_rewards += self.cost_gamma * torch.squeeze(
                self.transition.cost_values * time_outs, 1
            )
            self.transition.cost_rewards = self._sanitize_tensor(
                self.transition.cost_rewards, nan=0.0, posinf=1.0e3, neginf=0.0, clamp=1.0e3
            )

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
        mean_viol_loss = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

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
        ) in generator:

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )
                    if self.normalize_cost_advantage:
                        cost_advantages_batch = (
                            cost_advantages_batch - cost_advantages_batch.mean()
                        ) / (cost_advantages_batch.std() + 1e-8)

            advantages_batch = self._sanitize_tensor(
                advantages_batch, nan=0.0, posinf=1.0e3, neginf=-1.0e3, clamp=1.0e3
            )
            cost_advantages_batch = self._sanitize_tensor(
                cost_advantages_batch, nan=0.0, posinf=1.0e3, neginf=-1.0e3, clamp=1.0e3
            )
            returns_batch = self._sanitize_tensor(
                returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_returns_batch = self._sanitize_tensor(
                cost_returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            target_values_batch = self._sanitize_tensor(
                target_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_values_batch = self._sanitize_tensor(
                cost_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
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
                    kl = self._safe_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)
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
            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            cost_surrogate = (torch.squeeze(cost_advantages_batch) * ratio).mean()
            batch_cost_return, batch_cost_violation, c_hat = self._batch_cost_stats(
                cost_returns_batch
            )
            viol_loss = self._positive_cost_penalty(cost_surrogate, c_hat)
            surrogate_loss = self._sanitize_tensor(
                surrogate_loss, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
            )
            viol_loss = self._sanitize_tensor(viol_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6)

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
            value_loss = self._sanitize_tensor(value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6)

            # Cost value function loss
            if self.use_clipped_value_loss:
                cost_value_clipped = cost_values_batch + (
                    cost_value_batch - cost_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()
            cost_value_loss = self._sanitize_tensor(
                cost_value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            loss = (
                surrogate_loss
                + viol_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            loss = self._sanitize_tensor(loss, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6)
            if not torch.isfinite(loss):
                continue

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
            self._step_constraint_scale()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_viol_loss += viol_loss.item()
            mean_cost_return += batch_cost_return.item()
            mean_cost_violation += batch_cost_violation.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_viol_loss /= num_updates
        mean_cost_return /= num_updates
        mean_cost_violation /= num_updates
        # -- Clear the storage
        self.storage.clear()

        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": self.cost_limit - mean_cost_return,
            "cost_violation_rate": mean_cost_violation,
            "viol_loss": mean_viol_loss,
            "k_value": self.k_value,
        }

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
        }
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
        grads = [
            param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None
        ]
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
