from crl_isaaclab.terrains.crl_terrain_generator_cfg import CRLTerrainGeneratorCfg
from crl_isaaclab.terrains.crl_terrains import crl_terrains
from crl_isaaclab.terrains.crl_terrains.crl_terrains_cfg import * 

EXTREME_SAFELOCOMOTION_TERRAINS_CFG = CRLTerrainGeneratorCfg(
    # Align global terrain generator settings with FPPO rough-terrain configuration.
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    num_goals=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "crl_gap": ExtremeCRLGapTerrainCfg(
            proportion=0.5,
            apply_roughness=True,
            x_range=(0.8, 1.5),
            half_valid_width=(0.6, 1.2),
            gap_size="0.1 + 0.7*difficulty",
        ),
        "crl_step": ExtremeCRLStepTerrainCfg(
            proportion=0.5,
            apply_roughness=True,
            x_range=(0.3, 1.5),
            half_valid_width=(0.5, 1),
            step_height="0.1 + 0.35*difficulty",
        ),
    },
)
