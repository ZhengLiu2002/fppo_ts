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
    """FPPO with multi-constraint soft projection and conservative checks."""

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
        epsilon_safe=0.0,
        delta_kl: float | None = None,
        backtrack_coeff=0.5,
        max_corrections=10,
        projection_eps=1e-8,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        use_clipped_surrogate: bool = True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        k_decay: float = 1.0,
        k_min: float = 0.0,
        k_violation_threshold: float = 0.02,
        # Preconditioner/momentum
        use_preconditioner: bool = True,
        preconditioner_beta: float = 0.999,
        preconditioner_eps: float = 1.0e-8,
        use_momentum: bool = True,
        momentum_beta: float = 0.9,
        # Soft projection and conservative checks
        slack_penalty: float = 1.0,
        active_set_threshold: float = 0.05,
        confidence_level: float = 0.05,
        softproj_max_iters: int = 40,
        softproj_tol: float = 1.0e-6,
        constraint_limits: list[float] | tuple[float, ...] | None = None,
        # Compatibility knobs from previous FPPO variants
        feasible_first: bool = True,
        feasible_first_coef: float = 1.0,
        projection_scale_clip: float = 1.0e3,
        feasible_cost_margin: float = 1.0e-3,
        infeasible_improve_ratio: float = 0.01,
        infeasible_improve_abs: float = 1.0e-3,
        min_step_size: float = 1.0e-7,
        relax_cost_margin: float = 0.2,
        step_size_adaptive: bool = True,
        step_size_up: float = 1.02,
        step_size_down: float = 0.7,
        step_size_min: float = 5.0e-5,
        step_size_max: float = 2.0e-3,
        target_accept_rate: float = 0.75,
        step_size_cost_margin: float = 0.2,
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
            cost_limit=cost_limit,
            cost_viol_loss_coef=cost_viol_loss_coef,
            k_value=k_value,
            k_growth=k_growth,
            k_max=k_max,
            k_decay=k_decay,
            k_min=k_min,
            k_violation_threshold=k_violation_threshold,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.step_size = float(step_size)
        self.safe_radius = float(delta_safe)
        self.epsilon_safe = float(epsilon_safe)
        self.delta_kl = float(delta_kl) if delta_kl is not None else None
        self.backtrack_coeff = float(backtrack_coeff)
        self.max_corrections = int(max_corrections)
        self.projection_eps = float(projection_eps)
        self.use_clipped_surrogate = bool(use_clipped_surrogate)

        self.use_preconditioner = bool(use_preconditioner)
        self.preconditioner_beta = float(preconditioner_beta)
        self.preconditioner_eps = float(preconditioner_eps)
        self.use_momentum = bool(use_momentum)
        self.momentum_beta = float(momentum_beta)

        self.slack_penalty = max(float(slack_penalty), 1.0e-8)
        self.active_set_threshold = float(active_set_threshold)
        self.confidence_level = min(max(float(confidence_level), 1.0e-8), 1.0 - 1.0e-8)
        self.softproj_max_iters = max(int(softproj_max_iters), 1)
        self.softproj_tol = max(float(softproj_tol), 1.0e-12)
        self.constraint_limits = (
            torch.as_tensor(constraint_limits, dtype=torch.float32)
            if constraint_limits is not None
            else None
        )

        self.min_step_size = float(min_step_size)
        self.step_size_adaptive = bool(step_size_adaptive)
        self.step_size_up = float(step_size_up)
        self.step_size_down = float(step_size_down)
        self.step_size_min = float(step_size_min)
        self.step_size_max = float(step_size_max)
        self.target_accept_rate = float(target_accept_rate)
        self.step_size_cost_margin = float(step_size_cost_margin)

        # Keep these fields for compatibility with old configs.
        self.feasible_first = bool(feasible_first)
        self.feasible_first_coef = float(feasible_first_coef)
        self.projection_scale_clip = float(projection_scale_clip)
        self.feasible_cost_margin = float(feasible_cost_margin)
        self.infeasible_improve_ratio = float(infeasible_improve_ratio)
        self.infeasible_improve_abs = float(infeasible_improve_abs)
        self.relax_cost_margin = float(relax_cost_margin)

        critic_params = list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters())
        self.optimizer = {"critic": optim.Adam(critic_params, lr=learning_rate)}

        self._actor_params = self._get_actor_params()
        self._precond_v = None
        if self.use_preconditioner:
            self._precond_v = [torch.zeros_like(param, device=self.device) for param in self._actor_params]
        self._momentum = None
        if self.use_momentum:
            self._momentum = [torch.zeros_like(param, device=self.device) for param in self._actor_params]
        self.train_metrics: dict[str, float] = {}

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_viol_loss = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_cost_margin = 0.0
        mean_step_size = 0.0
        mean_kl = 0.0
        mean_active_constraints = 0.0
        accepted_updates = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        num_updates = self.num_learning_epochs * self.num_mini_batches

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
            *extra_batch,
        ) in generator:
            cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
            cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
            cost_term_samples_batch = extra_batch[2] if len(extra_batch) > 2 else None
            cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )
                    if self.normalize_cost_advantage:
                        cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (
                            cost_advantages_batch.std() + 1e-8
                        )
                        if cost_term_advantages_batch is not None:
                            mean = cost_term_advantages_batch.mean(dim=0, keepdim=True)
                            std = cost_term_advantages_batch.std(dim=0, keepdim=True)
                            cost_term_advantages_batch = (cost_term_advantages_batch - mean) / (
                                std + 1e-8
                            )

            advantages_batch = self._sanitize_tensor(
                advantages_batch, nan=0.0, posinf=1.0e3, neginf=-1.0e3, clamp=1.0e3
            )
            returns_batch = self._sanitize_tensor(
                returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_values_batch = self._sanitize_tensor(
                cost_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_returns_batch = self._sanitize_tensor(
                cost_returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            target_values_batch = self._sanitize_tensor(
                target_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )

            cost_terms_ret, cost_terms_adv, cost_terms_samples, cost_terms_val = self._prepare_cost_term_batches(
                cost_returns_batch=cost_returns_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_term_returns_batch=cost_term_returns_batch,
                cost_term_advantages_batch=cost_term_advantages_batch,
                cost_term_samples_batch=cost_term_samples_batch,
                cost_term_values_batch=cost_term_values_batch,
            )

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
            ratio_flat = ratio.reshape(-1)
            adv_flat = advantages_batch.reshape(-1)
            surrogate = -adv_flat * ratio_flat
            if self.use_clipped_surrogate:
                surrogate_clipped = -adv_flat * torch.clamp(
                    ratio_flat, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            else:
                surrogate_loss = surrogate.mean()
            policy_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

            # Reward gradient g_k
            g_list = torch.autograd.grad(
                policy_loss, self._actor_params, retain_graph=True, allow_unused=True
            )
            g_list = [
                (-grad if grad is not None else torch.zeros_like(param)).detach()
                for grad, param in zip(g_list, self._actor_params)
            ]
            if self.use_momentum and self._momentum is not None:
                with torch.no_grad():
                    for m, grad in zip(self._momentum, g_list):
                        m.mul_(self.momentum_beta).add_(grad, alpha=1.0 - self.momentum_beta)
                g_list = [m.detach() for m in self._momentum]

            # Per-constraint gradients a_{i,k}
            cost_surrogates = []
            cost_grad_lists = []
            num_constraints = cost_terms_adv.shape[1]
            for idx in range(num_constraints):
                adv_i = cost_terms_adv[:, idx]
                cost_obj_i = torch.mean(ratio_flat * adv_i)
                cost_surrogates.append(cost_obj_i.detach())
                grads_i = torch.autograd.grad(
                    cost_obj_i,
                    self._actor_params,
                    retain_graph=idx < (num_constraints - 1),
                    allow_unused=True,
                )
                grads_i = [
                    (grad if grad is not None else torch.zeros_like(param)).detach()
                    for grad, param in zip(grads_i, self._actor_params)
                ]
                cost_grad_lists.append(grads_i)

            if self.is_multi_gpu:
                self._all_reduce_grads(g_list)
                for grads_i in cost_grad_lists:
                    self._all_reduce_grads(grads_i)

            if self.use_preconditioner:
                self._update_preconditioner(g_list)
            p_vec = self._get_preconditioner_vector()
            g_vec = self._flatten_tensors(g_list)
            a_mat = torch.stack([self._flatten_tensors(grads_i) for grads_i in cost_grad_lists], dim=1)

            # b_{i,k} = d'_i - J_{C_i}(theta_k)
            d_limits = self._resolve_constraint_limits(num_constraints, device=cost_terms_ret.device)
            d_tight = d_limits - self.epsilon_safe
            j_cost = cost_terms_ret.mean(dim=0)
            b_budget = d_tight - j_cost

            theta_anchor = self._actor_param_vector().detach()
            alpha_k = self.step_size
            accepted = False
            final_kl = torch.as_tensor(0.0, device=self.device)
            final_ucb = torch.full_like(d_tight, float("inf"))
            final_active_count = 0
            delta_i = max(self.confidence_level / max(num_constraints, 1), 1.0e-8)

            # Conservative correction loop: nominal step -> soft projection -> KL -> per-constraint UCB.
            for _ in range(max(self.max_corrections, 1)):
                theta_nom = theta_anchor + alpha_k * p_vec * g_vec
                theta_proj, active_count = self._soft_project_lin_subset(
                    theta_nom=theta_nom,
                    theta_anchor=theta_anchor,
                    a_mat=a_mat,
                    b_budget=b_budget,
                    preconditioner=p_vec,
                )
                self._set_actor_param_vector(theta_proj)

                with torch.inference_mode():
                    self.policy.act(
                        obs_batch,
                        masks=masks_batch,
                        hidden_states=hid_states_batch[0],
                    )
                    mu_batch = self.policy.action_mean
                    sigma_batch = self.policy.action_std
                    final_kl = self._all_reduce_mean(
                        torch.mean(self._safe_kl(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch))
                    )
                    new_log_prob = self.policy.get_actions_log_prob(actions_batch)
                    ratio_candidate = self._safe_ratio(new_log_prob, old_actions_log_prob_batch).reshape(-1)

                kl_limit = self.delta_kl
                if kl_limit is None:
                    kl_limit = self.desired_kl if self.desired_kl is not None else float("inf")
                if final_kl > kl_limit:
                    self._set_actor_param_vector(theta_anchor)
                    alpha_k *= self.backtrack_coeff
                    if alpha_k < self.min_step_size:
                        break
                    continue

                feasible = True
                for idx in range(num_constraints):
                    ucb_i = self._clipped_is_cost_ucb(
                        ratio=ratio_candidate,
                        cost_samples=cost_terms_samples[:, idx],
                        clip_eps=self.clip_param,
                        delta_i=delta_i,
                    )
                    final_ucb[idx] = ucb_i
                    if ucb_i > d_tight[idx]:
                        feasible = False
                        break

                if feasible:
                    accepted = True
                    final_active_count = active_count
                    break

                self._set_actor_param_vector(theta_anchor)
                alpha_k *= self.backtrack_coeff
                if alpha_k < self.min_step_size:
                    break

            if not accepted:
                self._set_actor_param_vector(theta_anchor)
                alpha_k = 0.0
            else:
                accepted_updates += 1

            # Critic update (reward value + per-constraint cost-value heads)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[2]
            )
            cost_value_batch = self._sanitize_tensor(
                cost_value_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            if cost_value_batch.ndim == 1:
                cost_value_batch = cost_value_batch.unsqueeze(-1)
            elif cost_value_batch.ndim > 2:
                cost_value_batch = cost_value_batch.view(cost_value_batch.shape[0], -1)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            value_loss = self._sanitize_tensor(
                value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            pred_cost_terms = self._match_cost_heads(cost_value_batch, cost_terms_ret.shape[1])
            old_cost_terms = self._match_cost_heads(cost_terms_val, cost_terms_ret.shape[1])
            if self.use_clipped_value_loss:
                cost_value_clipped = old_cost_terms + (
                    pred_cost_terms - old_cost_terms
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (pred_cost_terms - cost_terms_ret).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_terms_ret).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_terms_ret - pred_cost_terms).pow(2).mean()
            cost_value_loss = self._sanitize_tensor(
                cost_value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            critic_loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
            critic_loss = self._sanitize_tensor(
                critic_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )
            self.optimizer["critic"].zero_grad()
            critic_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer["critic"].step()

            # Metrics
            batch_cost_return = self._all_reduce_mean(j_cost.mean())
            batch_cost_margin = self._all_reduce_mean(torch.min(d_tight - j_cost))
            batch_cost_violation = self._all_reduce_mean(
                (cost_terms_ret > d_limits.unsqueeze(0)).any(dim=1).float().mean()
            )
            c_hat = self._all_reduce_mean(torch.max(j_cost - d_tight))
            cost_surrogate_mean = torch.mean(torch.stack(cost_surrogates))
            viol_loss = self._positive_cost_penalty(cost_surrogate_mean, c_hat)
            viol_loss = self._sanitize_tensor(
                viol_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            self._step_constraint_scale(batch_cost_violation.item())
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_viol_loss += viol_loss.item()
            mean_cost_return += batch_cost_return.item()
            mean_cost_violation += batch_cost_violation.item()
            mean_cost_margin += batch_cost_margin.item()
            mean_step_size += alpha_k
            mean_kl += final_kl.item()
            mean_active_constraints += float(final_active_count)

        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_viol_loss /= num_updates
        mean_cost_return /= num_updates
        mean_cost_violation /= num_updates
        mean_cost_margin /= num_updates
        mean_step_size /= num_updates
        mean_kl /= num_updates
        mean_active_constraints /= num_updates
        accept_rate = accepted_updates / max(1, num_updates)
        reject_rate = 1.0 - accept_rate
        self._adapt_step_size(accept_rate=accept_rate, mean_cost_margin=mean_cost_margin)

        self.learning_rate = mean_step_size
        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": mean_cost_margin,
            "cost_violation_rate": mean_cost_violation,
            "viol_loss": mean_viol_loss,
            "k_value": self.k_value,
            "step_size": mean_step_size,
            "base_step_size": self.step_size,
            "accept_rate": accept_rate,
            "reject_rate": reject_rate,
            "kl": mean_kl,
            "active_constraints": mean_active_constraints,
        }

        self.storage.clear()
        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
        }

    def _prepare_cost_term_batches(
        self,
        cost_returns_batch: torch.Tensor,
        cost_advantages_batch: torch.Tensor,
        cost_term_returns_batch: torch.Tensor | None,
        cost_term_advantages_batch: torch.Tensor | None,
        cost_term_samples_batch: torch.Tensor | None,
        cost_term_values_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cost_term_returns_batch is None or cost_term_advantages_batch is None:
            fallback_samples = (
                cost_term_samples_batch if cost_term_samples_batch is not None else cost_returns_batch
            )
            fallback_values = (
                cost_term_values_batch if cost_term_values_batch is not None else cost_returns_batch
            )
            if not torch.is_tensor(fallback_samples):
                fallback_samples = torch.as_tensor(fallback_samples, device=self.device)
            if not torch.is_tensor(fallback_values):
                fallback_values = torch.as_tensor(fallback_values, device=self.device)
            fallback_samples = self._sanitize_tensor(
                fallback_samples.to(self.device),
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            fallback_values = self._sanitize_tensor(
                fallback_values.to(self.device),
                nan=0.0,
                posinf=1.0e4,
                neginf=-1.0e4,
                clamp=1.0e4,
            )
            if fallback_samples.ndim == 1:
                fallback_samples = fallback_samples.unsqueeze(-1)
            if fallback_values.ndim == 1:
                fallback_values = fallback_values.unsqueeze(-1)
            return (
                cost_returns_batch.reshape(-1, 1),
                cost_advantages_batch.reshape(-1, 1),
                fallback_samples.reshape(-1, fallback_samples.shape[-1]),
                fallback_values.reshape(-1, fallback_values.shape[-1]),
            )
        if not torch.is_tensor(cost_term_returns_batch):
            cost_term_returns_batch = torch.as_tensor(cost_term_returns_batch, device=self.device)
        if not torch.is_tensor(cost_term_advantages_batch):
            cost_term_advantages_batch = torch.as_tensor(cost_term_advantages_batch, device=self.device)
        cost_term_returns_batch = self._sanitize_tensor(
            cost_term_returns_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        cost_term_advantages_batch = self._sanitize_tensor(
            cost_term_advantages_batch.to(self.device),
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )
        if cost_term_returns_batch.ndim == 1:
            cost_term_returns_batch = cost_term_returns_batch.unsqueeze(-1)
        if cost_term_advantages_batch.ndim == 1:
            cost_term_advantages_batch = cost_term_advantages_batch.unsqueeze(-1)
        if cost_term_samples_batch is None:
            cost_term_samples_batch = cost_term_returns_batch
        if cost_term_values_batch is None:
            cost_term_values_batch = cost_term_returns_batch
        if not torch.is_tensor(cost_term_samples_batch):
            cost_term_samples_batch = torch.as_tensor(cost_term_samples_batch, device=self.device)
        if not torch.is_tensor(cost_term_values_batch):
            cost_term_values_batch = torch.as_tensor(cost_term_values_batch, device=self.device)
        cost_term_samples_batch = self._sanitize_tensor(
            cost_term_samples_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        cost_term_values_batch = self._sanitize_tensor(
            cost_term_values_batch.to(self.device),
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        if cost_term_samples_batch.ndim == 1:
            cost_term_samples_batch = cost_term_samples_batch.unsqueeze(-1)
        if cost_term_values_batch.ndim == 1:
            cost_term_values_batch = cost_term_values_batch.unsqueeze(-1)
        return (
            cost_term_returns_batch.reshape(-1, cost_term_returns_batch.shape[-1]),
            cost_term_advantages_batch.reshape(-1, cost_term_advantages_batch.shape[-1]),
            cost_term_samples_batch.reshape(-1, cost_term_samples_batch.shape[-1]),
            cost_term_values_batch.reshape(-1, cost_term_values_batch.shape[-1]),
        )

    @staticmethod
    def _match_cost_heads(cost_values: torch.Tensor, target_heads: int) -> torch.Tensor:
        if cost_values.ndim == 1:
            cost_values = cost_values.unsqueeze(-1)
        if cost_values.shape[-1] == target_heads:
            return cost_values
        if cost_values.shape[-1] == 1 and target_heads > 1:
            return cost_values.expand(-1, target_heads)
        if cost_values.shape[-1] > target_heads:
            return cost_values[:, :target_heads]
        pad = cost_values[:, -1:].expand(-1, target_heads - cost_values.shape[-1])
        return torch.cat([cost_values, pad], dim=-1)

    def _resolve_constraint_limits(self, num_constraints: int, device: torch.device) -> torch.Tensor:
        if self.constraint_limits is None:
            return torch.full((num_constraints,), float(self.cost_limit), device=device)
        d = self.constraint_limits.to(device=device, dtype=torch.float32)
        if d.numel() == 1:
            return d.expand(num_constraints)
        if d.numel() != num_constraints:
            return torch.full((num_constraints,), float(d.flatten()[0].item()), device=device)
        return d.flatten()

    def _soft_project_lin_subset(
        self,
        theta_nom: torch.Tensor,
        theta_anchor: torch.Tensor,
        a_mat: torch.Tensor,
        b_budget: torch.Tensor,
        preconditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delta_nom = theta_nom - theta_anchor
        if a_mat.numel() == 0:
            return theta_nom, 0
        violation = a_mat.transpose(0, 1).matmul(delta_nom) - b_budget
        active_mask = (violation > 0.0) | (b_budget < self.active_set_threshold)
        if not torch.any(active_mask):
            return theta_nom, 0
        a_active = a_mat[:, active_mask]
        v_active = violation[active_mask]
        q = a_active.transpose(0, 1).matmul(preconditioner.unsqueeze(1) * a_active)
        q = q + torch.eye(q.shape[0], device=q.device, dtype=q.dtype) / self.slack_penalty
        lamb = self._solve_nonnegative_qp(q, v_active)
        correction = preconditioner * (a_active.matmul(lamb))
        theta_proj = theta_nom - correction
        return theta_proj, int(active_mask.sum().item())

    def _soft_project_lin_active(
        self,
        theta_prime: torch.Tensor,
        theta_anchor: torch.Tensor,
        a_mat: torch.Tensor,
        b_budget: torch.Tensor,
        preconditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # Backward-compatible alias.
        return self._soft_project_lin_subset(
            theta_nom=theta_prime,
            theta_anchor=theta_anchor,
            a_mat=a_mat,
            b_budget=b_budget,
            preconditioner=preconditioner,
        )

    def _solve_nonnegative_qp(self, q_mat: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        if q_mat.numel() == 0:
            return torch.zeros_like(v_vec)
        max_eig = torch.linalg.eigvalsh(q_mat).max()
        step = 1.0 / (max_eig + self.projection_eps)
        lamb = torch.zeros_like(v_vec)
        for _ in range(self.softproj_max_iters):
            grad = q_mat.matmul(lamb) - v_vec
            new_lamb = torch.clamp(lamb - step * grad, min=0.0)
            if torch.max(torch.abs(new_lamb - lamb)) <= self.softproj_tol:
                lamb = new_lamb
                break
            lamb = new_lamb
        return lamb

    def _clipped_is_cost_ucb(
        self,
        ratio: torch.Tensor,
        cost_samples: torch.Tensor,
        clip_eps: float,
        delta_i: float,
    ) -> torch.Tensor:
        ratio = ratio.reshape(-1)
        cost_samples = cost_samples.reshape(-1).to(device=ratio.device, dtype=ratio.dtype)
        weight = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        x = weight * cost_samples

        if x.numel() == 0:
            return torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        n = torch.tensor(float(x.numel()), device=x.device, dtype=x.dtype)
        sum_x = x.sum()
        sum_x2 = torch.sum(x * x)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_x2, op=torch.distributed.ReduceOp.SUM)

        denom = torch.clamp(n, min=1.0)
        mu = sum_x / denom
        if n.item() > 1.0:
            var = torch.clamp((sum_x2 - n * mu * mu) / (n - 1.0), min=0.0)
        else:
            var = torch.zeros_like(mu)

        delta_i = min(max(float(delta_i), 1.0e-8), 1.0 - 1.0e-8)
        kappa = math.sqrt(2.0 * math.log(1.0 / delta_i))
        return mu + kappa * torch.sqrt(var / denom)

    def _adapt_step_size(self, accept_rate: float, mean_cost_margin: float):
        if not self.step_size_adaptive:
            return
        if mean_cost_margin > self.step_size_cost_margin and accept_rate >= self.target_accept_rate:
            self.step_size = min(self.step_size_max, self.step_size * self.step_size_up)
        elif mean_cost_margin < 0.0 or accept_rate < self.target_accept_rate * 0.5:
            self.step_size = max(self.step_size_min, self.step_size * self.step_size_down)

    def _step_constraint_scale(self, cost_violation: float | None = None):
        if cost_violation is None:
            super()._step_constraint_scale()
            return
        if cost_violation > self.k_violation_threshold:
            self.k_value = min(self.k_max, self.k_value * self.k_growth)
        else:
            self.k_value = max(self.k_min, self.k_value * self.k_decay)

    def _all_reduce_grads(self, grads):
        for grad in grads:
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            grad /= self.gpu_world_size

    def _get_actor_params(self):
        params = list(self.policy.actor.parameters())
        if hasattr(self.policy, "std"):
            params.append(self.policy.std)
        elif hasattr(self.policy, "log_std"):
            params.append(self.policy.log_std)
        return params

    def _flatten_tensors(self, tensor_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.reshape(-1) for tensor in tensor_list], dim=0)

    def _actor_param_vector(self) -> torch.Tensor:
        return self._flatten_tensors([param.data for param in self._actor_params])

    def _set_actor_param_vector(self, vector: torch.Tensor):
        offset = 0
        for param in self._actor_params:
            numel = param.numel()
            param.data.copy_(vector[offset : offset + numel].view_as(param))
            offset += numel

    def _update_preconditioner(self, grads):
        if not self.use_preconditioner or self._precond_v is None:
            return
        with torch.no_grad():
            for v, grad in zip(self._precond_v, grads):
                v.mul_(self.preconditioner_beta).addcmul_(grad, grad, value=1.0 - self.preconditioner_beta)

    def _get_preconditioner_vector(self) -> torch.Tensor:
        if not self.use_preconditioner or self._precond_v is None:
            return torch.ones_like(self._actor_param_vector())
        p_list = [1.0 / (torch.sqrt(v) + self.preconditioner_eps) for v in self._precond_v]
        return self._flatten_tensors(p_list)

    def _compute_kl(self, obs_batch, old_mu, old_sigma, masks_batch, hidden_states):
        with torch.no_grad():
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hidden_states)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            kl = self._safe_kl(mu_batch, sigma_batch, old_mu, old_sigma)
            return self._all_reduce_mean(torch.mean(kl))
