# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim

from .ppo import PPO


class FPPO(PPO):
    """First-Order Projected PPO for CMDPs."""

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
        step_size=1e-3,
        cost_limit=0.0,
        delta_safe=0.01,
        backtrack_coeff=0.5,
        max_backtracks=10,
        projection_eps=1e-8,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        use_clipped_surrogate: bool = True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
        use_preconditioner: bool = True,
        preconditioner_beta: float = 0.999,
        preconditioner_eps: float = 1e-8,
        feasible_first: bool = True,
        feasible_first_coef: float = 1.0,
        # VAE auxiliary parameters
        learning_rate_vae: float = 2.0e-3,
        vae_beta: float = 0.01,
        vae_beta_min: float = 1.0e-4,
        vae_beta_max: float = 0.1,
        vae_desired_recon_loss: float = 0.1,
        derived_action_loss_weight: float = 0.0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            cost_gamma=cost_gamma,
            cost_lam=cost_lam,
            value_loss_coef=value_loss_coef,
            cost_value_loss_coef=cost_value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            normalize_cost_advantage=normalize_cost_advantage,
            learning_rate_vae=learning_rate_vae,
            vae_beta=vae_beta,
            vae_beta_min=vae_beta_min,
            vae_beta_max=vae_beta_max,
            vae_desired_recon_loss=vae_desired_recon_loss,
            derived_action_loss_weight=derived_action_loss_weight,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.step_size = step_size
        self.cost_limit = cost_limit
        self.delta_safe = delta_safe
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks
        self.projection_eps = projection_eps
        self.use_clipped_surrogate = use_clipped_surrogate
        self.learning_rate = step_size
        self.critic_learning_rate = learning_rate
        self.use_preconditioner = use_preconditioner
        self.preconditioner_beta = preconditioner_beta
        self.preconditioner_eps = preconditioner_eps
        self.feasible_first = feasible_first
        self.feasible_first_coef = feasible_first_coef

        critic_params = list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters())
        self.optimizer = {"critic": optim.Adam(critic_params, lr=learning_rate)}
        # Optional VAE optimizer (auxiliary)
        self.vae_optimizer = None
        if getattr(self.policy, "vae_enabled", False) and getattr(self.policy, "vae", None) is not None:
            vae_lr = getattr(self.policy, "vae_learning_rate", None)
            if vae_lr is None:
                vae_lr = learning_rate
            self.vae_optimizer = optim.Adam(self.policy.vae.parameters(), lr=vae_lr)

        self._actor_params = self._get_actor_params()
        self._precond_v = None
        if self.use_preconditioner:
            self._precond_v = [torch.zeros_like(param, device=self.device) for param in self._actor_params]
        self.train_metrics: dict[str, float] = {}

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_step_size = 0.0
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

        num_updates = self.num_learning_epochs * self.num_mini_batches

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
            # Normalize advantages per mini-batch if requested.
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                    if self.normalize_cost_advantage:
                        cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (
                            cost_advantages_batch.std() + 1e-8
                        )

            # Actor forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_loss = surrogate
            if self.use_clipped_surrogate:
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped)
            surrogate_loss = surrogate_loss.mean()
            cost_surrogate = (torch.squeeze(cost_advantages_batch) * ratio).mean()
            policy_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

            # Compute reward and cost gradients for projection.
            g = torch.autograd.grad(policy_loss, self._actor_params, retain_graph=True)
            g = [(-gi).detach() for gi in g]
            g_c = torch.autograd.grad(cost_surrogate, self._actor_params, retain_graph=False)
            g_c = [gci.detach() for gci in g_c]

            if self.is_multi_gpu:
                self._all_reduce_grads(g)
                self._all_reduce_grads(g_c)

            dot_gc_g = self._sum_grads_product(g_c, g)
            gc_norm_sq = self._sum_grads_product(g_c, g_c)

            cost_return_mean = cost_returns_batch.mean()
            if self.is_multi_gpu:
                cost_return_mean = self._all_reduce_mean(cost_return_mean)
            c_hat = cost_return_mean - self.cost_limit
            if self.feasible_first and c_hat > 0 and dot_gc_g.item() > 0.0:
                g = [gi - self.feasible_first_coef * gci for gi, gci in zip(g, g_c)]
                dot_gc_g = self._sum_grads_product(g_c, g)

            inv_precond = None
            if self.use_preconditioner:
                self._update_preconditioner(g)
                inv_precond = self._get_inv_preconditioner()

            base_params = [param.data.clone() for param in self._actor_params]
            alpha = self.step_size
            kl_mean = torch.tensor(0.0, device=self.device)

            if self.delta_safe is None or self.max_backtracks <= 0:
                v = self._project_direction(g, g_c, dot_gc_g, gc_norm_sq, c_hat, alpha, inv_precond)
                self._apply_update(self._actor_params, base_params, v, alpha)
                kl_mean = self._compute_kl(obs_batch, old_mu_batch, old_sigma_batch, masks_batch, hid_states_batch[0])
            else:
                for _ in range(self.max_backtracks + 1):
                    v = self._project_direction(g, g_c, dot_gc_g, gc_norm_sq, c_hat, alpha, inv_precond)
                    self._apply_update(self._actor_params, base_params, v, alpha)
                    kl_mean = self._compute_kl(
                        obs_batch, old_mu_batch, old_sigma_batch, masks_batch, hid_states_batch[0]
                    )
                    if kl_mean <= self.delta_safe:
                        break
                    alpha *= self.backtrack_coeff

            # Critic updates
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[2]
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.use_clipped_value_loss:
                cost_value_clipped = cost_values_batch + (cost_value_batch - cost_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            critic_loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
            self.optimizer["critic"].zero_grad()
            critic_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters()), self.max_grad_norm
            )
            self.optimizer["critic"].step()

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

            # Book keeping
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl += kl_mean.item()
            mean_cost_return += cost_return_mean.item()
            violation_rate = (cost_returns_batch > self.cost_limit).float().mean()
            if self.is_multi_gpu:
                violation_rate = self._all_reduce_mean(violation_rate)
            mean_cost_violation += violation_rate.item()
            mean_step_size += alpha

        # -- Aggregate
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_kl /= num_updates
        mean_cost_return /= num_updates
        mean_cost_violation /= num_updates
        mean_step_size /= num_updates
        if num_updates > 0 and self.vae_optimizer is not None:
            mean_vae_vel_loss /= num_updates
            mean_vae_mass_loss /= num_updates
            mean_vae_decode_loss /= num_updates
            mean_vae_kl_loss /= num_updates
            mean_vae_beta /= num_updates
            mean_derived_action_loss /= num_updates

        self.learning_rate = mean_step_size
        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": self.cost_limit - mean_cost_return,
            "cost_violation_rate": mean_cost_violation,
            "kl": mean_kl,
            "step_size": mean_step_size,
        }

        # -- Clear the storage
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            **(
                {
                    "vae_vel_loss": mean_vae_vel_loss,
                    "vae_mass_loss": mean_vae_mass_loss,
                    "vae_decode_loss": mean_vae_decode_loss,
                    "vae_kl_loss": mean_vae_kl_loss,
                    "vae_beta": mean_vae_beta,
                    "derived_action_loss": mean_derived_action_loss,
                }
                if self.vae_optimizer is not None
                else {}
            ),
        }

    def _get_actor_params(self):
        params = list(self.policy.actor.parameters())
        if hasattr(self.policy, "std"):
            params.append(self.policy.std)
        elif hasattr(self.policy, "log_std"):
            params.append(self.policy.log_std)
        return params

    def _project_direction(self, g, g_c, dot_gc_g, gc_norm_sq, c_hat, alpha, inv_precond=None):
        if inv_precond is None:
            gc_norm_value = gc_norm_sq.item()
        else:
            gc_norm_value = sum((gci * inv_i * gci).sum() for gci, inv_i in zip(g_c, inv_precond)).item()
        if gc_norm_value < self.projection_eps:
            return g
        b_value = (-c_hat / alpha).item()
        dot_value = dot_gc_g.item()
        if dot_value <= b_value:
            return g
        scale = (dot_value - b_value) / (gc_norm_value + self.projection_eps)
        if inv_precond is None:
            return [gi - scale * gci for gi, gci in zip(g, g_c)]
        return [gi - scale * inv_i * gci for gi, gci, inv_i in zip(g, g_c, inv_precond)]

    def _apply_update(self, params, base_params, direction, alpha):
        for param, base, direction_i in zip(params, base_params, direction):
            param.data.copy_(base + alpha * direction_i)

    def _sum_grads_product(self, grads_a, grads_b):
        return sum((ga * gb).sum() for ga, gb in zip(grads_a, grads_b))

    def _update_preconditioner(self, grads):
        if not self.use_preconditioner or self._precond_v is None:
            return
        with torch.no_grad():
            for v, g in zip(self._precond_v, grads):
                v.mul_(self.preconditioner_beta).addcmul_(g, g, value=1.0 - self.preconditioner_beta)

    def _get_inv_preconditioner(self):
        if not self.use_preconditioner or self._precond_v is None:
            return None
        inv = []
        for v in self._precond_v:
            inv.append(1.0 / (torch.sqrt(v) + self.preconditioner_eps))
        return inv

    def _all_reduce_grads(self, grads):
        for grad in grads:
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            grad /= self.gpu_world_size

    def _all_reduce_mean(self, value):
        if self.is_multi_gpu:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    def _compute_kl(self, obs_batch, old_mu, old_sigma, masks_batch, hidden_states):
        with torch.no_grad():
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hidden_states)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma + 1.0e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu_batch)) / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            return kl_mean
