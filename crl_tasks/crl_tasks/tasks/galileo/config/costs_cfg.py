"""Cost (constraint) configuration for Galileo CRL tasks.

Deprecated: cost parameters now live in mdp_cfg.py.
This module re-exports the cost configs for backwards compatibility.
"""

from __future__ import annotations

from .mdp_cfg import LEG_JOINT_CFG, CostsCfg, StudentCostsCfg, TeacherCostsCfg

__all__ = [
    "LEG_JOINT_CFG",
    "TeacherCostsCfg",
    "StudentCostsCfg",
    "CostsCfg",
]
