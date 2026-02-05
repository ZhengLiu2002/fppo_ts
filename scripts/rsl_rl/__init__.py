"""Lightweight package init to avoid eager Omni/IsaacSim imports."""

from .cli_args import *  # noqa: F401,F403

try:
    from .exporter import (  # noqa: F401
        export_teacher_policy_as_jit,
        export_teacher_policy_as_onnx,
    )
except Exception:
    # Keep imports lazy to avoid requiring Isaac Sim outside runtime.
    pass
