# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .omnisafe_utils import (
    conjugate_gradients,
    flatten_tensor_sequence,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_parameters,
    trainable_parameters,
)
from .ppo import PPO


class CPO(PPO):
    """Constrained Policy Optimization ported from OmniSafe's second-order update."""

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
        backtrack_coeff=0.8,
        max_backtracks=20,
        projection_eps=1e-8,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
        cost_viol_loss_coef: float = 0.0,
        k_value: float = 1.0,
        k_growth: float = 1.0,
        k_max: float = 1.0,
        cg_iters: int = 10,
        cg_damping: float = 1e-2,
        fvp_sample_freq: int = 1,
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
        self.step_size = float(step_size)
        self.delta_safe = delta_safe
        self.backtrack_coeff = float(backtrack_coeff)
        self.max_backtracks = int(max_backtracks)
        self.projection_eps = float(projection_eps)
        self.cg_iters = max(int(cg_iters), 1)
        self.cg_damping = float(cg_damping)
        self.fvp_sample_freq = max(int(fvp_sample_freq), 1)
        self.target_kl = float(desired_kl) if desired_kl is not None else 0.01

        self._policy_parameters = self._collect_policy_parameters()
        self._value_parameters = self._collect_value_parameters()
        self.value_optimizer = optim.Adam(self._value_parameters, lr=learning_rate)
        self.optimizer = self.value_optimizer
        self.train_metrics: dict[str, float] = {}

    def _collect_policy_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.policy.actor.parameters())
        std_param = getattr(self.policy, "std", None)
        log_std_param = getattr(self.policy, "log_std", None)
        if isinstance(std_param, torch.nn.Parameter):
            params.append(std_param)
        if isinstance(log_std_param, torch.nn.Parameter):
            params.append(log_std_param)
        return trainable_parameters(params)

    def _collect_value_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        params.extend(self.policy.critic.parameters())
        params.extend(self.policy.cost_critic.parameters())
        critic_scan_encoder = getattr(self.policy, "_critic_scan_encoder", None)
        if critic_scan_encoder is not None:
            params.extend(critic_scan_encoder.parameters())
        unique_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in params:
            if id(parameter) in seen or not parameter.requires_grad:
                continue
            unique_params.append(parameter)
            seen.add(id(parameter))
        if not unique_params:
            raise RuntimeError("CPO requires critic parameters for value updates.")
        return unique_params

    def _make_distribution(self, obs_batch: torch.Tensor) -> torch.distributions.Normal:
        self.policy.act(obs_batch)
        mean = self._sanitize_tensor(
            self.policy.action_mean,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        std = self._sanitize_tensor(
            self.policy.action_std,
            nan=1.0e-6,
            posinf=1.0e2,
            neginf=1.0e-6,
            clamp=1.0e2,
        ).clamp_min(1.0e-6)
        return torch.distributions.Normal(mean, std)

    def _mean_kl(
        self,
        old_distribution: torch.distributions.Normal,
        new_distribution: torch.distributions.Normal,
    ) -> torch.Tensor:
        kl = torch.distributions.kl_divergence(old_distribution, new_distribution).sum(dim=-1).mean()
        return self._all_reduce_mean(kl)

    def _loss_pi_reward(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_log_prob_batch: torch.Tensor,
        reward_advantages: torch.Tensor,
    ) -> torch.Tensor:
        self._make_distribution(obs_batch)
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        ratio = self._safe_ratio(actions_log_prob_batch, old_log_prob_batch)
        return -(ratio * reward_advantages).mean()

    def _loss_pi_cost(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_log_prob_batch: torch.Tensor,
        cost_advantages: torch.Tensor,
    ) -> torch.Tensor:
        self._make_distribution(obs_batch)
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        ratio = self._safe_ratio(actions_log_prob_batch, old_log_prob_batch)
        return (ratio * cost_advantages).mean()

    def _fisher_vector_product(
        self,
        obs_batch: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        sample_obs = obs_batch[:: self.fvp_sample_freq]
        self.policy.zero_grad(set_to_none=True)
        q_dist = self._make_distribution(sample_obs)
        with torch.no_grad():
            p_dist = torch.distributions.Normal(q_dist.mean.detach(), q_dist.stddev.detach())
        kl = torch.distributions.kl_divergence(p_dist, q_dist).sum(dim=-1).mean()
        grads = torch.autograd.grad(
            kl,
            tuple(self._policy_parameters),
            create_graph=True,
            allow_unused=True,
        )
        flat_grad_kl = flatten_tensor_sequence(grads, self._policy_parameters)
        kl_p = torch.dot(flat_grad_kl, vector)
        hvp = torch.autograd.grad(
            kl_p,
            tuple(self._policy_parameters),
            retain_graph=False,
            allow_unused=True,
        )
        flat_hvp = flatten_tensor_sequence(hvp, self._policy_parameters)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(flat_hvp, op=torch.distributed.ReduceOp.SUM)
            flat_hvp /= self.gpu_world_size
        return flat_hvp + self.cg_damping * vector

    def _get_full_batch(self):
        if self.policy.is_recurrent:
            raise NotImplementedError("CPO/PCPO currently support feed-forward policies only.")
        return next(self.storage.mini_batch_generator(1, 1))

    def _prepare_actor_batch(self) -> dict[str, torch.Tensor]:
        (
            obs_batch,
            _critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            _returns_batch,
            _cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            *extra_batch,
        ) = self._get_full_batch()

        cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
        cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
        cost_terms_ret, cost_terms_adv, _ = self._prepare_cost_term_batches(
            cost_returns_batch=cost_returns_batch,
            cost_advantages_batch=cost_advantages_batch,
            cost_term_returns_batch=cost_term_returns_batch,
            cost_term_advantages_batch=cost_term_advantages_batch,
            cost_term_values_batch=extra_batch[3] if len(extra_batch) > 3 else None,
        )
        aggregate_cost_returns = torch.sum(cost_terms_ret, dim=1)
        aggregate_cost_advantages = torch.sum(cost_terms_adv, dim=1)
        batch_cost_return, batch_cost_violation, c_hat = self._batch_cost_stats(aggregate_cost_returns)

        reward_advantages = self._sanitize_tensor(
            torch.squeeze(advantages_batch),
            nan=0.0,
            posinf=1.0e3,
            neginf=-1.0e3,
            clamp=1.0e3,
        )
        old_mu_batch = self._sanitize_tensor(
            old_mu_batch,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
            clamp=1.0e4,
        )
        old_sigma_batch = self._sanitize_tensor(
            old_sigma_batch,
            nan=1.0e-6,
            posinf=1.0e2,
            neginf=1.0e-6,
            clamp=1.0e2,
        ).clamp_min(1.0e-6)

        return {
            "obs": obs_batch,
            "actions": actions_batch,
            "old_logp": old_actions_log_prob_batch,
            "reward_adv": reward_advantages,
            "cost_adv": aggregate_cost_advantages,
            "old_dist": torch.distributions.Normal(old_mu_batch, old_sigma_batch),
            "cost_return": batch_cost_return,
            "cost_violation": batch_cost_violation,
            "c_hat": c_hat,
        }

    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        old_distribution: torch.distributions.Normal,
        actor_batch: dict[str, torch.Tensor],
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        total_steps: int,
        decay: float,
        violation_c: torch.Tensor,
        optim_case: int,
    ) -> tuple[torch.Tensor, int, float]:
        step_frac = 1.0
        theta_old = get_flat_params_from(self._policy_parameters)
        expected_reward_improve = grads.dot(step_direction)
        final_kl = torch.zeros((), device=self.device)
        total_steps = max(int(total_steps), 1)
        violation_value = float(violation_c.item())

        for step in range(total_steps):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_parameters(self._policy_parameters, new_theta)
            acceptance_step = step + 1

            with torch.no_grad():
                loss_reward = self._loss_pi_reward(
                    actor_batch["obs"],
                    actor_batch["actions"],
                    actor_batch["old_logp"],
                    actor_batch["reward_adv"],
                )
                loss_cost = self._loss_pi_cost(
                    actor_batch["obs"],
                    actor_batch["actions"],
                    actor_batch["old_logp"],
                    actor_batch["cost_adv"],
                )
                q_dist = self._make_distribution(actor_batch["obs"])
                kl = self._mean_kl(old_distribution, q_dist)

            loss_reward_improve = self._all_reduce_mean(loss_reward_before - loss_reward)
            loss_cost_diff = self._all_reduce_mean(loss_cost - loss_cost_before)

            if not torch.isfinite(loss_reward) or not torch.isfinite(loss_cost):
                pass
            elif not torch.isfinite(kl):
                pass
            elif optim_case > 1 and loss_reward_improve < 0:
                pass
            elif loss_cost_diff > max(-violation_value, 0.0):
                pass
            elif kl > self.target_kl:
                pass
            else:
                final_kl = kl
                break
            step_frac *= decay
        else:
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        set_param_values_to_parameters(self._policy_parameters, theta_old)
        _ = expected_reward_improve
        return step_frac * step_direction, acceptance_step, float(final_kl.item())

    def _determine_case(
        self,
        b_grads: torch.Tensor,
        ep_costs: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            return 4, torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        A = q - r**2 / (s + 1e-8)
        B = 2 * self.target_kl - ep_costs**2 / (s + 1e-8)
        if ep_costs < 0 and B < 0:
            optim_case = 3
        elif ep_costs < 0 <= B:
            optim_case = 2
        elif ep_costs >= 0 and B >= 0:
            optim_case = 1
        else:
            optim_case = 0
        return optim_case, A, B

    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        ep_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1, device=self.device)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x
        elif optim_case in (1, 2):
            lambda_a = torch.sqrt(torch.clamp_min(A / (B + 1e-8), 0.0))
            lambda_b = torch.sqrt(torch.clamp_min(q / (2 * self.target_kl + 1e-8), 0.0))
            r_num = r.item()
            inf_tensor = torch.tensor(float("inf"), device=self.device)
            zero_tensor = torch.tensor(0.0, device=self.device)
            boundary = torch.tensor(r_num, device=self.device) / (ep_costs + 1e-8)
            if ep_costs < 0:
                lambda_a_star = torch.clamp(lambda_a, zero_tensor, boundary)
                lambda_b_star = torch.clamp(lambda_b, boundary, inf_tensor)
            else:
                lambda_a_star = torch.clamp(lambda_a, boundary, inf_tensor)
                lambda_b_star = torch.clamp(lambda_b, zero_tensor, boundary)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )
            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0.0) / (s + 1e-8)
            step_direction = (x - nu_star * p) / (lambda_star + 1e-8)
        else:
            lambda_star = torch.zeros(1, device=self.device)
            nu_star = torch.sqrt(2 * self.target_kl / (s + 1e-8))
            step_direction = -nu_star * p
        return step_direction, lambda_star, nu_star

    def _compute_step_direction(
        self,
        xHx: torch.Tensor,
        x: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        ep_costs: torch.Tensor,
        b_grads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        optim_case, A, B = self._determine_case(b_grads, ep_costs, q, r, s)
        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )
        return step_direction, lambda_star, nu_star, optim_case, A, B

    def _update_actor(self) -> dict[str, float]:
        actor_batch = self._prepare_actor_batch()
        theta_old = get_flat_params_from(self._policy_parameters)

        self.policy.zero_grad(set_to_none=True)
        loss_reward = self._loss_pi_reward(
            actor_batch["obs"],
            actor_batch["actions"],
            actor_batch["old_logp"],
            actor_batch["reward_adv"],
        )
        loss_reward_before = loss_reward.detach()
        loss_reward.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        grads = -get_flat_gradients_from(self._policy_parameters)

        x = conjugate_gradients(
            lambda vec: self._fisher_vector_product(actor_batch["obs"], vec),
            grads,
            num_steps=self.cg_iters,
        )
        xHx = torch.dot(x, self._fisher_vector_product(actor_batch["obs"], x))
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))

        self.policy.zero_grad(set_to_none=True)
        loss_cost = self._loss_pi_cost(
            actor_batch["obs"],
            actor_batch["actions"],
            actor_batch["old_logp"],
            actor_batch["cost_adv"],
        )
        loss_cost_before = loss_cost.detach()
        loss_cost.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        b_grads = get_flat_gradients_from(self._policy_parameters)

        p = conjugate_gradients(
            lambda vec: self._fisher_vector_product(actor_batch["obs"], vec),
            b_grads,
            num_steps=self.cg_iters,
        )
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)
        step_direction, lambda_star, nu_star, optim_case, A, B = self._compute_step_direction(
            xHx=xHx,
            x=x,
            p=p,
            q=q,
            r=r,
            s=s,
            ep_costs=actor_batch["c_hat"],
            b_grads=b_grads,
        )

        step_direction, acceptance_step, final_kl = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            old_distribution=actor_batch["old_dist"],
            actor_batch=actor_batch,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=self.max_backtracks,
            decay=self.backtrack_coeff,
            violation_c=actor_batch["c_hat"],
            optim_case=optim_case,
        )
        theta_new = theta_old + step_direction
        set_param_values_to_parameters(self._policy_parameters, theta_new)

        with torch.no_grad():
            reward_loss = self._loss_pi_reward(
                actor_batch["obs"],
                actor_batch["actions"],
                actor_batch["old_logp"],
                actor_batch["reward_adv"],
            )
            cost_loss = self._loss_pi_cost(
                actor_batch["obs"],
                actor_batch["actions"],
                actor_batch["old_logp"],
                actor_batch["cost_adv"],
            )
            _ = self._make_distribution(actor_batch["obs"])
            entropy = self.policy.entropy.mean().item()

        return {
            "surrogate": float(reward_loss.item()),
            "cost_surrogate": float(cost_loss.item()),
            "entropy": float(entropy),
            "mean_cost_return": float(actor_batch["cost_return"].item()),
            "cost_limit_margin": float(self.cost_limit - actor_batch["cost_return"].item()),
            "cost_violation_rate": float(actor_batch["cost_violation"].item()),
            "kl": final_kl,
            "acceptance_step": float(acceptance_step),
            "alpha": float(alpha.item()),
            "final_step_norm": float(step_direction.norm().item()),
            "gradient_norm": float(torch.norm(grads).item()),
            "cost_gradient_norm": float(torch.norm(b_grads).item()),
            "lambda_star": float(lambda_star.item()),
            "nu_star": float(nu_star.item()),
            "optim_case": float(optim_case),
            "A": float(A.item()),
            "B": float(B.item()),
            "q": float(q.item()),
            "r": float(r.item()),
            "s": float(s.item()),
        }

    def _update_value_functions(self) -> tuple[float, float]:
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        num_updates = self.num_learning_epochs * self.num_mini_batches

        if self.policy.is_recurrent:
            raise NotImplementedError("CPO/PCPO currently support feed-forward policies only.")
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            _obs_batch,
            critic_obs_batch,
            _actions_batch,
            target_values_batch,
            _advantages_batch,
            returns_batch,
            cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            hid_states_batch,
            masks_batch,
            *extra_batch,
        ) in generator:
            cost_term_returns_batch = extra_batch[0] if len(extra_batch) > 0 else None
            cost_term_advantages_batch = extra_batch[1] if len(extra_batch) > 1 else None
            cost_term_values_batch = extra_batch[3] if len(extra_batch) > 3 else None

            returns_batch = self._sanitize_tensor(
                returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            target_values_batch = self._sanitize_tensor(
                target_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_values_batch = self._sanitize_tensor(
                cost_values_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_returns_batch = self._sanitize_tensor(
                cost_returns_batch, nan=0.0, posinf=1.0e4, neginf=-1.0e4, clamp=1.0e4
            )
            cost_terms_ret, _cost_terms_adv, cost_terms_val = self._prepare_cost_term_batches(
                cost_returns_batch=cost_returns_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_term_returns_batch=cost_term_returns_batch,
                cost_term_advantages_batch=cost_term_advantages_batch,
                cost_term_values_batch=cost_term_values_batch,
            )

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
            pred_cost_terms = self._match_cost_heads(cost_value_batch, cost_terms_ret.shape[1])
            old_cost_terms = self._match_cost_heads(cost_terms_val, cost_terms_ret.shape[1])

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
                cost_value_clipped = old_cost_terms + (
                    pred_cost_terms - old_cost_terms
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (pred_cost_terms - cost_terms_ret).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_terms_ret).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_terms_ret - pred_cost_terms).pow(2).mean()

            loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss
            self.policy.zero_grad(set_to_none=True)
            self.value_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self._value_parameters, self.max_grad_norm)
            self.value_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()

        return mean_value_loss / num_updates, mean_cost_value_loss / num_updates

    def update(self):
        if self.policy.is_recurrent:
            raise NotImplementedError("CPO/PCPO currently support feed-forward policies only.")

        actor_metrics = self._update_actor()
        mean_value_loss, mean_cost_value_loss = self._update_value_functions()

        self.storage.clear()
        self.train_metrics = actor_metrics
        return {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": actor_metrics["surrogate"],
            "cost_surrogate": actor_metrics["cost_surrogate"],
            "entropy": actor_metrics["entropy"],
        }
