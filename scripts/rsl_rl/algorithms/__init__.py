# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .cpo import CPO
from .distillation import Distillation
from .focpo import FOCPO
from .fppo import FPPO
from .pcpo import PCPO
from .ppo import PPO
from .ppo_lagrange import PPOLagrange

ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "fppo": FPPO,
    "ppo_lagrange": PPOLagrange,
    "cpo": CPO,
    "pcpo": PCPO,
    "focpo": FOCPO,
    "distillation": Distillation,
}

__all__ = ["PPO", "FPPO", "PPOLagrange", "CPO", "PCPO", "FOCPO", "Distillation", "ALGORITHM_REGISTRY"]
