from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_with_encoder import ActorCriticRMA
from .feature_extractors import DefaultEstimator
from scripts.rsl_rl.storage import RolloutStorage


class FPPOWithExtractor:
    """First-Order Projected PPO with privileged-state estimator."""

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
        # Constraint normalization knobs (handled by runner, accepted for compatibility)
        constraint_normalization: bool = True,
        constraint_norm_beta: float = 0.99,
        constraint_norm_min_scale: float = 1e-3,
        constraint_norm_max_scale: float = 10.0,
        constraint_norm_clip: float = 5.0,
        constraint_proxy_delta: float = 0.1,
        constraint_agg_tau: float = 0.5,
        constraint_scale_by_gamma: bool = False,
        constraint_cost_scale: float | None = None,
        # Estimator params
        priv_reg_coef_schedual=None,
        # Unused optional configs (kept for compatibility)
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy: ActorCriticRMA = policy.to(self.device)
        self.estimator: DefaultEstimator = estimator.to(self.device)

        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

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
        self.use_clipped_surrogate = use_clipped_surrogate
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.normalize_cost_advantage = normalize_cost_advantage
        self.learning_rate = learning_rate

        # FPPO-specific
        self.step_size = step_size
        self.cost_limit = cost_limit
        self.delta_safe = delta_safe
        self.backtrack_coeff = backtrack_coeff
        self.max_backtracks = max_backtracks
        self.projection_eps = projection_eps
        self.use_preconditioner = use_preconditioner
        self.preconditioner_beta = preconditioner_beta
        self.preconditioner_eps = preconditioner_eps
        self.feasible_first = feasible_first
        self.feasible_first_coef = feasible_first_coef

        # Critic optimizer (reward + cost)
        critic_params = list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters())
        self.optimizer = optim.Adam(critic_params, lr=learning_rate)

        # Estimator / history encoder optimizers
        self.priv_states_dim = estimator_paras["num_priv_explicit"]
        num_priv_hurdles = int(estimator_paras.get("num_priv_hurdles", 0))
        self.num_priv_hurdles = num_priv_hurdles
        self.priv_states_dim_other = max(int(self.priv_states_dim) - int(self.num_priv_hurdles), 0)
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        estimator_params = list(self.estimator.parameters())
        self.estimator_optimizer = (
            optim.Adam(estimator_params, lr=estimator_paras["learning_rate"]) if estimator_params else None
        )
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        hist_params = list(self.policy.actor.history_encoder.parameters())
        self.hist_encoder_optimizer = (
            optim.Adam(hist_params, lr=learning_rate) if len(hist_params) > 0 else None
        )

        self.priv_reg_coef_schedual = priv_reg_coef_schedual or [0.0, 0.1, 2000.0, 3000.0]
        self.counter = 0

        # No MoE/gating in FPPO-TS

        # Optional extras (not used in current configs)
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None

        # Actor params for projection
        self._actor_params = self._get_actor_params()
        self._precond_v = None
        if self.use_preconditioner:
            self._precond_v = [torch.zeros_like(param, device=self.device) for param in self._actor_params]

        self.train_metrics: dict[str, float] = {}

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, privileged_obs=None, actions_shape=None):
        if actions_shape is None:
            actions_shape = (self.policy.actor.num_actions,) if hasattr(self.policy, "actor") else (1,)
        if isinstance(actions_shape, list):
            actions_shape = tuple(actions_shape)
        obs_shape = obs if isinstance(obs, (list, tuple)) else (obs,)
        priv_shape = privileged_obs if privileged_obs and privileged_obs[0] > 0 else None
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            priv_shape,
            actions_shape,
            self.device,
        )

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, : self.num_prop])
            start = self.num_prop + self.num_scan + self.num_priv_hurdles
            end = self.num_prop + self.num_scan + self.priv_states_dim
            if self.priv_states_dim_other > 0:
                obs_est[:, start:end] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.policy.act(obs, hist_encoding).detach()

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, infos, costs=None):
        if self.storage is None:
            return
        self.transition.rewards = rewards.clone()
        if costs is None:
            costs = torch.zeros_like(rewards)
        self.transition.cost_rewards = costs.clone()
        self.transition.dones = dones
        self.transition.next_actor_observations = obs

        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.cost_rewards += self.cost_gamma * torch.squeeze(self.transition.cost_values * time_outs, 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, critic_obs):
        if self.storage is None:
            return
        last_values = self.policy.evaluate(critic_obs).detach()
        last_cost_values = self.policy.evaluate_cost(critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            last_cost_values=last_cost_values,
            cost_gamma=self.cost_gamma,
            cost_lam=self.cost_lam,
            normalize_cost_advantage=self.normalize_cost_advantage and not self.normalize_advantage_per_mini_batch,
        )

    def update(self):  # noqa: C901
        if self.storage is None:
            return {}

        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_cost_return = 0.0
        mean_cost_violation = 0.0
        mean_step_size = 0.0
        mean_priv_reg_loss = 0.0
        mean_estimator_loss = 0.0

        generator = (
            self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            if self.policy.is_recurrent
            else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        )

        num_updates = self.num_learning_epochs * self.num_mini_batches

        for batch in generator:
            (
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
                _actor_obs_batch,
                _vae_obs_history_batch,
                _next_actor_obs_batch,
                _amp_obs_batch,
            ) = batch

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

            # Privileged / history regularization
            priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(
                max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1
            )
            priv_reg_coef = (
                priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0])
                + self.priv_reg_coef_schedual[0]
            )

            # Estimator loss
            priv_states_predicted = self.estimator(obs_batch[:, : self.num_prop])
            start = self.num_prop + self.num_scan + self.num_priv_hurdles
            end = self.num_prop + self.num_scan + self.priv_states_dim
            if self.priv_states_dim_other > 0:
                estimator_loss = (priv_states_predicted - obs_batch[:, start:end]).pow(2).mean()
            else:
                estimator_loss = torch.zeros((), device=self.device)

            # Surrogate loss
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
            policy_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean() + priv_reg_coef * priv_reg_loss


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
            self.optimizer.zero_grad()
            critic_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                list(self.policy.critic.parameters()) + list(self.policy.cost_critic.parameters()), self.max_grad_norm
            )
            self.optimizer.step()

            # Estimator update (separate)
            if self.estimator_optimizer is not None:
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_estimator_grads()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()

            # Book keeping
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_cost_return += cost_return_mean.item()
            violation_rate = (cost_returns_batch > self.cost_limit).float().mean()
            if self.is_multi_gpu:
                violation_rate = self._all_reduce_mean(violation_rate)
            mean_cost_violation += violation_rate.item()
            mean_step_size += alpha
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_estimator_loss += estimator_loss.item()

        # Aggregate
        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_cost_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            mean_cost_return /= num_updates
            mean_cost_violation /= num_updates
            mean_step_size /= num_updates
            mean_priv_reg_loss /= num_updates
            mean_estimator_loss /= num_updates
        self.train_metrics = {
            "mean_cost_return": mean_cost_return,
            "cost_limit_margin": self.cost_limit - mean_cost_return,
            "cost_violation_rate": mean_cost_violation,
            "step_size": mean_step_size,
        }

        # Clear storage
        self.storage.clear()
        self.update_counter()

        loss_dict = {
            "value_function": mean_value_loss,
            "cost_value_function": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "mean_cost_return": mean_cost_return,
            "cost_violation_rate": mean_cost_violation,
            "priv_reg": mean_priv_reg_loss,
            "estimator": mean_estimator_loss,
        }
        return loss_dict

    def update_dagger(self):
        if self.storage is None:
            return 0.0
        mean_hist_latent_loss = 0.0
        generator = (
            self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            if self.policy.is_recurrent
            else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        )
        for batch in generator:
            obs_batch = batch[0]
            if obs_batch is None:
                continue
            with torch.inference_mode():
                self.policy.act(obs_batch, hist_encoding=True, masks=batch[13], hidden_states=batch[12][0])
            with torch.inference_mode():
                priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            if self.hist_encoder_optimizer is not None:
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if num_updates > 0:
            mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_counter(self):
        self.counter += 1

    def broadcast_parameters(self):
        if not self.is_multi_gpu:
            return
        state_to_sync = {
            "policy": self.policy.state_dict(),
            "estimator": self.estimator.state_dict(),
        }
        obj_list = [state_to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced_state = obj_list[0]
        self.policy.load_state_dict(synced_state["policy"])
        self.estimator.load_state_dict(synced_state["estimator"])

    def reduce_parameters(self):
        if not self.is_multi_gpu:
            return
        modules = [self.policy]
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                    param.grad /= self.gpu_world_size

    def _reduce_estimator_grads(self):
        if not self.is_multi_gpu:
            return
        for param in self.estimator.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad /= self.gpu_world_size

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
