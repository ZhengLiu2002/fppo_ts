# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.cost_rewards = None
            self.dones = None
            self.values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.cost_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.mu = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.sigma = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_returns = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.cost_advantages = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.saved_hidden_states_cost = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding new transitions."
            )

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        if transition.cost_rewards is not None:
            self.cost_rewards[self.step].copy_(transition.cost_rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.cost_values[self.step].copy_(transition.cost_values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None:
            return
        if len(hidden_states) == 2:
            hid_a_raw, hid_c_raw = hidden_states
            hid_cost_raw = None
        elif len(hidden_states) == 3:
            hid_a_raw, hid_c_raw, hid_cost_raw = hidden_states
        else:
            raise ValueError(
                "Hidden states must be a tuple of length 2 or 3 (actor, critic, optional cost critic)."
            )
        if hid_a_raw is None and hid_c_raw is None and hid_cost_raw is None:
            return
        # make a tuple out of GRU hidden state to match the LSTM format
        hid_a = hid_a_raw if isinstance(hid_a_raw, tuple) else (hid_a_raw,)
        hid_c = hid_c_raw if isinstance(hid_c_raw, tuple) else (hid_c_raw,)
        hid_cost = (
            None
            if hid_cost_raw is None
            else (hid_cost_raw if isinstance(hid_cost_raw, tuple) else (hid_cost_raw,))
        )
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                for i in range(len(hid_c))
            ]
            if hid_cost is not None:
                self.saved_hidden_states_cost = [
                    torch.zeros(self.observations.shape[0], *hid_cost[i].shape, device=self.device)
                    for i in range(len(hid_cost))
                ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])
        if hid_cost is not None and self.saved_hidden_states_cost is not None:
            for i in range(len(hid_cost)):
                self.saved_hidden_states_cost[i][self.step].copy_(hid_cost[i])

    def clear(self):
        self.step = 0

    def compute_returns(
        self,
        last_values,
        gamma,
        lam,
        normalize_advantage: bool = True,
        last_cost_values=None,
        cost_gamma=None,
        cost_lam=None,
        normalize_cost_advantage: bool = True,
    ):
        advantage = 0
        cost_advantage = 0
        cost_gamma = gamma if cost_gamma is None else cost_gamma
        cost_lam = lam if cost_lam is None else cost_lam
        compute_cost = self.training_type == "rl" and last_cost_values is not None

        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_cost_values = last_cost_values if compute_cost else None
            else:
                next_values = self.values[step + 1]
                next_cost_values = self.cost_values[step + 1] if compute_cost else None
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            )
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

            if compute_cost and next_cost_values is not None:
                cost_delta = (
                    self.cost_rewards[step]
                    + next_is_not_terminal * cost_gamma * next_cost_values
                    - self.cost_values[step]
                )
                cost_advantage = (
                    cost_delta + next_is_not_terminal * cost_gamma * cost_lam * cost_advantage
                )
                self.cost_returns[step] = cost_advantage + self.cost_values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1e-8
            )

        if compute_cost:
            self.cost_advantages = self.cost_returns - self.cost_values
            if normalize_cost_advantage:
                self.cost_advantages = (self.cost_advantages - self.cost_advantages.mean()) / (
                    self.cost_advantages.std() + 1e-8
                )

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[
                i
            ], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                cost_values_batch = cost_values[batch_idx]
                cost_returns_batch = cost_returns[batch_idx]
                cost_advantages_batch = cost_advantages[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield (
                    obs_batch,
                    privileged_observations_batch,
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
                    (None, None, None),
                    None,
                )

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # split trajectories into episode segments
        trajectories = split_and_pad_trajectories(self.dones, self.observations)
        # (T, B, ...)
        obs_traj = trajectories[0]
        traj_masks = trajectories[1]

        # split trajectories into episode segments for other tensors
        if self.privileged_observations is not None:
            privileged_obs_traj = split_and_pad_trajectories(
                self.dones, self.privileged_observations
            )[0]
        else:
            privileged_obs_traj = obs_traj
        actions_traj = split_and_pad_trajectories(self.dones, self.actions)[0]
        values_traj = split_and_pad_trajectories(self.dones, self.values)[0]
        returns_traj = split_and_pad_trajectories(self.dones, self.returns)[0]
        cost_values_traj = split_and_pad_trajectories(self.dones, self.cost_values)[0]
        cost_returns_traj = split_and_pad_trajectories(self.dones, self.cost_returns)[0]
        cost_advantages_traj = split_and_pad_trajectories(self.dones, self.cost_advantages)[0]
        old_actions_log_prob_traj = split_and_pad_trajectories(self.dones, self.actions_log_prob)[0]
        advantages_traj = split_and_pad_trajectories(self.dones, self.advantages)[0]
        old_mu_traj = split_and_pad_trajectories(self.dones, self.mu)[0]
        old_sigma_traj = split_and_pad_trajectories(self.dones, self.sigma)[0]

        # RNN hidden states
        if self.saved_hidden_states_a is None:
            raise RuntimeError("Hidden states not available for recurrent policy.")

        # Flatten trajectory batch dimension
        batch_size = obs_traj.shape[1]
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = obs_traj[:, batch_idx]
                privileged_obs_batch = privileged_obs_traj[:, batch_idx]
                actions_batch = actions_traj[:, batch_idx]
                target_values_batch = values_traj[:, batch_idx]
                returns_batch = returns_traj[:, batch_idx]
                cost_values_batch = cost_values_traj[:, batch_idx]
                cost_returns_batch = cost_returns_traj[:, batch_idx]
                cost_advantages_batch = cost_advantages_traj[:, batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob_traj[:, batch_idx]
                advantages_batch = advantages_traj[:, batch_idx]
                old_mu_batch = old_mu_traj[:, batch_idx]
                old_sigma_batch = old_sigma_traj[:, batch_idx]

                # hidden states: (num_layers, batch, hidden_dim)
                hid_a = self.saved_hidden_states_a[:, batch_idx]
                hid_c = self.saved_hidden_states_c[:, batch_idx]
                hid_cost = None
                if self.saved_hidden_states_cost is not None:
                    hid_cost = self.saved_hidden_states_cost[:, batch_idx]

                yield (
                    obs_batch,
                    privileged_obs_batch,
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
                    (hid_a, hid_c, hid_cost),
                    traj_masks[:, batch_idx],
                )
