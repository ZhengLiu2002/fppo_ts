# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .registry import (
    get_algorithm_class,
    list_algorithm_names,
    list_algorithm_aliases,
    register_algorithm,
)

PPO = get_algorithm_class("ppo")
FPPO = get_algorithm_class("fppo")
NP3O = get_algorithm_class("np3o")
PPOLagrange = get_algorithm_class("ppo_lagrange")
CPO = get_algorithm_class("cpo")
PCPO = get_algorithm_class("pcpo")
FOCPO = get_algorithm_class("focpo")
Distillation = get_algorithm_class("distillation")

# Backward-compatible mapping (lower-case keys).
ALGORITHM_REGISTRY = {name: get_algorithm_class(name) for name in list_algorithm_names()}

__all__ = [
    "PPO",
    "FPPO",
    "NP3O",
    "PPOLagrange",
    "CPO",
    "PCPO",
    "FOCPO",
    "Distillation",
    "ALGORITHM_REGISTRY",
    "get_algorithm_class",
    "list_algorithm_names",
    "list_algorithm_aliases",
    "register_algorithm",
]
