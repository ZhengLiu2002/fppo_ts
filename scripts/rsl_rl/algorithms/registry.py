"""Algorithm registry and config validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
import os
import warnings


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata for an RL algorithm implementation."""

    name: str
    module: str
    class_name: str
    training_type: str = "rl"
    aliases: tuple[str, ...] = ()
    extra_cfg_keys: tuple[str, ...] = ()

    def load_class(self):
        module = import_module(self.module)
        try:
            return getattr(module, self.class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Algorithm class '{self.class_name}' not found in module '{self.module}'."
            ) from exc


_REGISTRY: dict[str, AlgorithmSpec] = {}
_CANONICAL: dict[str, AlgorithmSpec] = {}


def _normalize(name: str) -> str:
    return name.strip().lower()


def register_algorithm(spec: AlgorithmSpec) -> None:
    """Register an algorithm spec.

    Args:
        spec: AlgorithmSpec describing the implementation.
    """
    canonical_key = _normalize(spec.name)
    if canonical_key in _CANONICAL:
        raise ValueError(f"Algorithm '{spec.name}' already registered.")
    _CANONICAL[canonical_key] = spec
    for alias in (spec.name, spec.class_name, *spec.aliases):
        _REGISTRY[_normalize(alias)] = spec


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    key = _normalize(name)
    if key not in _REGISTRY:
        available = ", ".join(sorted(list_algorithm_names()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return _REGISTRY[key]


def get_algorithm_class(name: str):
    spec = get_algorithm_spec(name)
    return spec.load_class()


def get_algorithm_class_name(name: str) -> str:
    return get_algorithm_spec(name).class_name


def list_algorithm_names() -> list[str]:
    return sorted(spec.name for spec in _CANONICAL.values())


def list_algorithm_aliases() -> list[str]:
    return sorted(_REGISTRY.keys())


def algorithm_allowed_keys(alg_class) -> set[str]:
    sig = inspect.signature(alg_class.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    allowed.discard("policy")
    return allowed


def validate_algorithm_cfg(
    alg_class,
    cfg: dict,
    *,
    extra_allowed_keys: set[str] | None = None,
    strict: bool = False,
) -> None:
    extra_allowed_keys = extra_allowed_keys or set()
    allowed = algorithm_allowed_keys(alg_class) | extra_allowed_keys | {"class_name"}
    unknown = sorted(set(cfg.keys()) - allowed)
    if not unknown:
        return
    msg = (
        "Algorithm config contains unused keys: "
        f"{unknown}. Allowed keys for {alg_class.__name__}: {sorted(allowed)}"
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)


def strict_algorithm_cfg_enabled() -> bool:
    return os.getenv("CRL_ALGO_STRICT", "0") == "1"


# Default registry
register_algorithm(
    AlgorithmSpec(
        name="ppo",
        module="scripts.rsl_rl.algorithms.ppo",
        class_name="PPO",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="fppo",
        module="scripts.rsl_rl.algorithms.fppo",
        class_name="FPPO",
        training_type="rl",
        extra_cfg_keys=("priv_reg_coef_schedual",),
    )
)
register_algorithm(
    AlgorithmSpec(
        name="np3o",
        module="scripts.rsl_rl.algorithms.np3o",
        class_name="NP3O",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="ppo_lagrange",
        module="scripts.rsl_rl.algorithms.ppo_lagrange",
        class_name="PPOLagrange",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="cpo",
        module="scripts.rsl_rl.algorithms.cpo",
        class_name="CPO",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="pcpo",
        module="scripts.rsl_rl.algorithms.pcpo",
        class_name="PCPO",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="focpo",
        module="scripts.rsl_rl.algorithms.focpo",
        class_name="FOCPO",
        training_type="rl",
    )
)
register_algorithm(
    AlgorithmSpec(
        name="distillation",
        module="scripts.rsl_rl.algorithms.distillation",
        class_name="Distillation",
        training_type="distillation",
    )
)

__all__ = [
    "AlgorithmSpec",
    "register_algorithm",
    "get_algorithm_spec",
    "get_algorithm_class",
    "get_algorithm_class_name",
    "list_algorithm_names",
    "list_algorithm_aliases",
    "algorithm_allowed_keys",
    "validate_algorithm_cfg",
    "strict_algorithm_cfg_enabled",
]
