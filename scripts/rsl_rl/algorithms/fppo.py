# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
        # 安全相关：参数空间安全半径、成本安全裕度
        delta_safe=0.01,  # reinterpret as parameter-space safe radius (Δ_safe)
        epsilon_safe=0.0,  # cost tightening margin
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
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        use_preconditioner: bool = True,
        use_momentum: bool = True,
        momentum_beta: float = 0.9,
        preconditioner_beta: float = 0.999,
        preconditioner_eps: float = 1e-8,
        slack_penalty: float = 1.0,
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
        k_decay: float = 1.0,
        k_min: float = 0.0,
        k_violation_threshold: float = 0.02,
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
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.step_size = step_size
        # reinterpret delta_safe as parameter-space safe radius Δ_safe (norm on update)
        self.safe_radius = delta_safe
        self.epsilon_safe = epsilon_safe
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks
        self.projection_eps = projection_eps
        self.use_clipped_surrogate = use_clipped_surrogate
        self.learning_rate = step_size
        self.critic_learning_rate = learning_rate
        self.use_preconditioner = use_preconditioner
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta
        self.preconditioner_beta = preconditioner_beta
        self.preconditioner_eps = preconditioner_eps
        self.slack_penalty = slack_penalty
        self.feasible_first = feasible_first
        self.feasible_first_coef = feasible_first_coef
        self.projection_scale_clip = projection_scale_clip
        self.feasible_cost_margin = float(feasible_cost_margin)
        self.infeasible_improve_ratio = float(infeasible_improve_ratio)
        self.infeasible_improve_abs = float(infeasible_improve_abs)
        self.min_step_size = float(min_step_size)
        self.relax_cost_margin = float(relax_cost_margin)
        self.step_size_adaptive = bool(step_size_adaptive)
        self.step_size_up = float(step_size_up)
        self.step_size_down = float(step_size_down)
        self.step_size_min = float(step_size_min)
        self.step_size_max = float(step_size_max)
        self.target_accept_rate = float(target_accept_rate)
        self.step_size_cost_margin = float(step_size_cost_margin)
        self.k_decay = float(k_decay)
        self.k_min = float(k_min)
        self.k_violation_threshold = float(k_violation_threshold)

        critic_params = list(self.policy.critic.parameters()) + list(
            self.policy.cost_critic.parameters()
        )
        self.optimizer = {"critic": optim.Adam(critic_params, lr=learning_rate)}

        self._actor_params = self._get_actor_params()
        self._precond_v = None
        if self.use_preconditioner:
            self._precond_v = [
                torch.zeros_like(param, device=self.device) for param in self._actor_params
            ]
        self._momentum = None
        if self.use_momentum:
            self._momentum = [
                torch.zeros_like(param, device=self.device) for param in self._actor_params
            ]
        self.train_metrics: dict[str, float] = {}

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_viol_loss = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_step_size = 0.0
        accepted_updates = 0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

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
            *_,
        ) in generator:
            # Normalize advantages per mini-batch if requested.
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

            # Actor forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            ratio = self._safe_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
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
            surrogate_loss = self._sanitize_tensor(
                surrogate_loss, nan=0.0, posinf=1.0e6, neginf=-1.0e6, clamp=1.0e6
            )

            # Compute reward and cost gradients for projection.
            g = torch.autograd.grad(policy_loss, self._actor_params, retain_graph=True)
            g = [(-gi).detach() for gi in g]
            if self.use_momentum and self._momentum is not None:
                with torch.no_grad():
                    for m, gi in zip(self._momentum, g):
                        m.mul_(self.momentum_beta).add_(gi, alpha=1 - self.momentum_beta)
                g = [m.detach() for m in self._momentum]
            g_c = torch.autograd.grad(cost_surrogate, self._actor_params, retain_graph=False)
            g_c = [gci.detach() for gci in g_c]

            if self.is_multi_gpu:
                self._all_reduce_grads(g)
                self._all_reduce_grads(g_c)

            dot_gc_g = self._sum_grads_product(g_c, g)
            gc_norm_sq = self._sum_grads_product(g_c, g_c)

            cost_return_mean, batch_cost_violation, _ = self._batch_cost_stats(cost_returns_batch)
            # tightened cost limit: d' = d - epsilon_safe
            effective_cost_limit = self.cost_limit - self.epsilon_safe
            effective_cost_limit_t = torch.as_tensor(
                effective_cost_limit, device=cost_return_mean.device, dtype=cost_return_mean.dtype
            )
            c_hat = cost_return_mean - effective_cost_limit_t
            c_hat_proj = self.k_value * c_hat
            cost_slack = effective_cost_limit_t - cost_return_mean
            relax_projection = cost_slack.item() >= self.relax_cost_margin
            if self.feasible_first and c_hat_proj.item() > 0.0 and dot_gc_g.item() > 0.0:
                g = [gi - self.feasible_first_coef * gci for gi, gci in zip(g, g_c)]
                dot_gc_g = self._sum_grads_product(g_c, g)
            viol_loss = self._positive_cost_penalty(cost_surrogate, c_hat)
            viol_loss = self._sanitize_tensor(
                viol_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

            inv_precond = None
            if self.use_preconditioner:
                self._update_preconditioner(g)
                inv_precond = self._get_inv_preconditioner()

            base_params = [param.data.clone() for param in self._actor_params]
            alpha = self.step_size

            # Backtracking line search with soft projection + KL/cost checks
            accepted = False
            for _ in range(self.max_backtracks + 1):
                # 1) Nominal PPO step
                self._apply_update(self._actor_params, base_params, g, alpha)
                if not relax_projection:
                    # 2) Soft projection onto linearized cost constraint
                    #    Compute violation v = A^T (theta'-theta_k) - b, where b = d' - J_C
                    diff = [p.data - b for p, b in zip(self._actor_params, base_params)]
                    dot_gc_diff = self._sum_grads_product(g_c, diff)
                    v_scalar = dot_gc_diff - (effective_cost_limit - cost_return_mean)
                    if self.use_preconditioner and inv_precond is not None:
                        q_scalar = (
                            self._sum_grads_product(
                                [gci * invi for gci, invi in zip(g_c, inv_precond)], g_c
                            )
                            + 1.0 / self.slack_penalty
                        )
                    else:
                        q_scalar = gc_norm_sq + 1.0 / self.slack_penalty
                    lambda_star = torch.clamp(v_scalar / (q_scalar + self.projection_eps), min=0.0)
                    if self.use_preconditioner and inv_precond is not None:
                        proj_direction = [
                            gci * invi * lambda_star for gci, invi in zip(g_c, inv_precond)
                        ]
                    else:
                        proj_direction = [gci * lambda_star for gci in g_c]
                    for param, proj_dir in zip(self._actor_params, proj_direction):
                        param.data.add_(-proj_dir)

                # 3) Acceptance: KL and cost check (importance weighted, clipped)
                with torch.inference_mode():
                    kl_mean = self._compute_kl(
                        obs_batch, old_mu_batch, old_sigma_batch, masks_batch, hid_states_batch[0]
                    )
                    kl_mean = self._sanitize_tensor(
                        kl_mean, nan=1.0e6, posinf=1.0e6, neginf=0.0, clamp=1.0e6
                    )
                    kl_limit = self.desired_kl if self.desired_kl is not None else float("inf")
                    if relax_projection:
                        cost_check = cost_return_mean
                        cost_accept_limit = cost_return_mean + self.feasible_cost_margin
                    else:
                        # IS-clipped cost estimate
                        self.policy.act(
                            obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                        new_log_prob = self.policy.get_actions_log_prob(actions_batch)
                        ratio = self._safe_ratio(new_log_prob, old_actions_log_prob_batch)
                        ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        cost_check = (ratio.unsqueeze(-1) * cost_returns_batch).mean()
                        cost_check = self._sanitize_tensor(
                            cost_check, nan=1.0e6, posinf=1.0e6, neginf=0.0, clamp=1.0e6
                        )

                        # Infeasible phase: require monotonic cost reduction.
                        if cost_return_mean.item() <= effective_cost_limit_t.item():
                            cost_accept_limit = effective_cost_limit_t + self.feasible_cost_margin
                        else:
                            improve_rel = torch.abs(cost_return_mean) * self.infeasible_improve_ratio
                            improve_abs = torch.as_tensor(
                                self.infeasible_improve_abs,
                                device=cost_return_mean.device,
                                dtype=cost_return_mean.dtype,
                            )
                            required_improve = torch.maximum(improve_rel, improve_abs)
                            cost_accept_limit = cost_return_mean - required_improve

                if kl_mean <= kl_limit and cost_check <= cost_accept_limit:
                    accepted = True
                    break

                # backtrack: restore and shrink step
                for param, base in zip(self._actor_params, base_params):
                    param.data.copy_(base)
                alpha *= self.backtrack_coeff
                if alpha < self.min_step_size:
                    break

            if not accepted:
                # Revert if no acceptable step found
                for param, base in zip(self._actor_params, base_params):
                    param.data.copy_(base)
                alpha = 0.0
            else:
                accepted_updates += 1

            # Critic updates
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
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
            value_loss = self._sanitize_tensor(
                value_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
            )

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

            critic_loss = (
                self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
            )
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

            # Book keeping
            self._step_constraint_scale(batch_cost_violation.item())
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_viol_loss += viol_loss.item()
            mean_cost_return += cost_return_mean.item()
            mean_cost_violation += batch_cost_violation.item()
            mean_step_size += alpha

        # -- Aggregate
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_viol_loss /= num_updates
        mean_cost_return /= num_updates
        mean_cost_violation /= num_updates
        mean_step_size /= num_updates
        accept_rate = accepted_updates / max(1, num_updates)
        reject_rate = 1.0 - accept_rate
        self._adapt_step_size(accept_rate, mean_cost_return)

        self.learning_rate = mean_step_size
        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": self.cost_limit - mean_cost_return,
            "cost_violation_rate": mean_cost_violation,
            "viol_loss": mean_viol_loss,
            "k_value": self.k_value,
            "step_size": mean_step_size,
            "base_step_size": self.step_size,
            "accept_rate": accept_rate,
            "reject_rate": reject_rate,
        }

        # -- Clear the storage
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "viol": mean_viol_loss,
        }

    def _step_constraint_scale(self, cost_violation: float | None = None):
        if cost_violation is None:
            super()._step_constraint_scale()
            return
        if cost_violation > self.k_violation_threshold:
            self.k_value = min(self.k_max, self.k_value * self.k_growth)
        else:
            self.k_value = max(self.k_min, self.k_value * self.k_decay)

    def _adapt_step_size(self, accept_rate: float, mean_cost_return: float):
        if not self.step_size_adaptive:
            return
        effective_limit = self.cost_limit - self.epsilon_safe
        if (
            mean_cost_return <= effective_limit - self.step_size_cost_margin
            and accept_rate >= self.target_accept_rate
        ):
            self.step_size = min(self.step_size_max, self.step_size * self.step_size_up)
        elif mean_cost_return > effective_limit or accept_rate < self.target_accept_rate * 0.5:
            self.step_size = max(self.step_size_min, self.step_size * self.step_size_down)

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
            gc_norm_value = sum(
                (gci * inv_i * gci).sum() for gci, inv_i in zip(g_c, inv_precond)
            ).item()
        if gc_norm_value < self.projection_eps:
            return g
        b_value = (-c_hat / alpha).item()
        dot_value = dot_gc_g.item()
        if dot_value <= b_value:
            return g
        scale = (dot_value - b_value) / (gc_norm_value + self.projection_eps)
        if self.projection_scale_clip is not None and self.projection_scale_clip > 0:
            scale = max(-self.projection_scale_clip, min(self.projection_scale_clip, scale))
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
                v.mul_(self.preconditioner_beta).addcmul_(
                    g, g, value=1.0 - self.preconditioner_beta
                )

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
                + (torch.square(old_sigma) + torch.square(old_mu - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            return kl_mean
