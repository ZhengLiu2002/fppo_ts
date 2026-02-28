# Algorithms

This repo standardizes algorithm integration via a registry and config validation.

## Registry

Algorithms are registered in:
- `scripts/rsl_rl/algorithms/registry.py`

Each algorithm declares:
- `name`: canonical CLI name (e.g. `fppo`)
- `module`: module path (e.g. `scripts.rsl_rl.algorithms.fppo`)
- `class_name`: class to instantiate (e.g. `FPPO`)
- `training_type`: `rl` or `distillation`
- `extra_cfg_keys`: optional keys allowed beyond the class `__init__` signature

## Validation

Before instantiation, algorithm configs are validated against the algorithm
constructor signature. Unknown keys raise warnings by default. To make it strict:

```bash
export CRL_ALGO_STRICT=1
```

## Adding a New Algorithm

1. Create the algorithm implementation in `scripts/rsl_rl/algorithms/`.
2. Register it in `scripts/rsl_rl/algorithms/registry.py` via `register_algorithm`.
3. Add default hyperparameters in
   `crl_tasks/crl_tasks/tasks/galileo/config/defaults.py` under `GalileoDefaults.algorithm`.
4. Use `--algo <name>` or set `GalileoDefaults.algorithm.name` to select it.

This keeps CLI choices, config validation, and runner behavior in sync.
