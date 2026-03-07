# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from .cpo import CPO


class PCPO(CPO):
    """Projection-based CPO update ported from OmniSafe PCPO."""

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
        _ = (xHx, b_grads)
        q_safe = torch.clamp_min(q, 1.0e-8)
        s_safe = torch.clamp_min(s, 1.0e-8)
        scale = torch.sqrt(2 * self.target_kl / q_safe)
        correction = torch.clamp_min((scale * r + ep_costs) / s_safe, 0.0)
        step_direction = scale * x - correction * p
        one = torch.ones(1, device=self.device)
        return step_direction, one, one, 1, one, one
