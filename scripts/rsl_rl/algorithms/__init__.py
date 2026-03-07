# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public algorithm namespace with lazy class resolution."""

from .registry import (
    get_algorithm_class,
    list_algorithm_aliases,
    list_algorithm_names,
    register_algorithm,
)

_ALGORITHM_EXPORTS = {
    "PPO": "ppo",
    "FPPO": "fppo",
    "NP3O": "np3o",
    "PPOLagrange": "ppo_lagrange",
    "CPO": "cpo",
    "PCPO": "pcpo",
    "FOCOPS": "focops",
    "Distillation": "distillation",
}


def __getattr__(name: str):
    if name == "ALGORITHM_REGISTRY":
        value = {algo_name: get_algorithm_class(algo_name) for algo_name in list_algorithm_names()}
        globals()[name] = value
        return value

    if name in _ALGORITHM_EXPORTS:
        value = get_algorithm_class(_ALGORITHM_EXPORTS[name])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PPO",
    "FPPO",
    "NP3O",
    "PPOLagrange",
    "CPO",
    "PCPO",
    "FOCOPS",
    "Distillation",
    "ALGORITHM_REGISTRY",
    "get_algorithm_class",
    "list_algorithm_names",
    "list_algorithm_aliases",
    "register_algorithm",
]
