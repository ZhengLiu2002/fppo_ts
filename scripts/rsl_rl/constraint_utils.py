# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Dict

import torch


class ConstraintNormalizer:
    """Normalize and aggregate per-term constraint costs."""

    def __init__(
        self,
        enabled: bool = True,
        ema_beta: float = 0.99,
        min_scale: float = 1e-3,
        max_scale: float = 10.0,
        clip: float = 5.0,
        huber_delta: float = 0.1,
        agg_tau: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.enabled = enabled
        self.ema_beta = ema_beta
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.clip = clip
        self.huber_delta = huber_delta
        self.agg_tau = agg_tau
        self.device = torch.device(device)
        self._scales: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_cfg(cls, cfg: dict, device: str | torch.device) -> "ConstraintNormalizer":
        return cls(
            enabled=cfg.get("constraint_normalization", True),
            ema_beta=cfg.get("constraint_norm_beta", 0.99),
            min_scale=cfg.get("constraint_norm_min_scale", 1e-3),
            max_scale=cfg.get("constraint_norm_max_scale", 10.0),
            clip=cfg.get("constraint_norm_clip", 5.0),
            huber_delta=cfg.get("constraint_proxy_delta", 0.1),
            agg_tau=cfg.get("constraint_agg_tau", 0.5),
            device=device,
        )

    def _as_tensor(self, value: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        value = value.to(self.device)
        if value.ndim > 1:
            if value.shape[-1] == 1:
                value = value.squeeze(-1)
            else:
                value = value.sum(dim=-1)
        return value

    def _apply_huber(self, value: torch.Tensor) -> torch.Tensor:
        if self.huber_delta <= 0.0:
            return value
        delta_t = torch.as_tensor(self.huber_delta, device=value.device, dtype=value.dtype)
        quad = 0.5 * (value**2) / delta_t
        lin = value - 0.5 * delta_t
        return torch.where(value < delta_t, quad, lin)

    def _update_scale(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_mean = value.mean()
            prev = self._scales.get(name)
            if prev is None:
                scale = batch_mean
            else:
                scale = self.ema_beta * prev + (1.0 - self.ema_beta) * batch_mean
            scale = torch.clamp(scale, min=self.min_scale, max=self.max_scale)
            self._scales[name] = scale
        return self._scales[name]

    def normalize(self, cost_terms: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
        if not self.enabled or not cost_terms:
            return {}
        normalized: dict[str, torch.Tensor] = {}
        for name, value in cost_terms.items():
            value_t = self._as_tensor(value)
            value_t = torch.clamp(value_t, min=0.0)
            value_t = self._apply_huber(value_t)
            scale = self._update_scale(name, value_t)
            norm = value_t / scale
            norm = torch.clamp(norm, min=0.0, max=self.clip)
            normalized[name] = norm
        return normalized

    def aggregate(self, cost_terms: dict[str, torch.Tensor | float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not cost_terms:
            return torch.zeros(0, device=self.device), {}
        normalized = self.normalize(cost_terms)
        if not normalized:
            first = self._as_tensor(next(iter(cost_terms.values())))
            return torch.zeros_like(first), {}
        names = sorted(normalized.keys())
        stacked = torch.stack([normalized[name] for name in names], dim=-1)
        tau = max(self.agg_tau, 1e-6)
        weights = torch.softmax(stacked / tau, dim=-1)
        agg = torch.sum(weights * stacked, dim=-1)
        return agg, normalized
