# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .fppo import FPPO


class CPO(FPPO):
    """Constrained Policy Optimization using first-order projection."""

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
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        normalize_cost_advantage: bool = False,
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
            step_size=step_size,
            cost_limit=cost_limit,
            delta_safe=delta_safe,
            backtrack_coeff=backtrack_coeff,
            max_backtracks=max_backtracks,
            projection_eps=projection_eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_clipped_surrogate=False,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            normalize_cost_advantage=normalize_cost_advantage,
            multi_gpu_cfg=multi_gpu_cfg,
        )
