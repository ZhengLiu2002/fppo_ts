from isaaclab.utils import configclass
from ..crl_terrain_generator_cfg import CRLSubTerrainBaseCfg
from . import crl_terrains

@configclass
class ExtremeCRLRoughTerrainCfg(CRLSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0

@configclass
class ExtremeCRLGapTerrainCfg(ExtremeCRLRoughTerrainCfg):
    function = crl_terrains.crl_gap_terrain
    gap_size: str = '0.1 + 0.7*difficulty'
    gap_depth: tuple[float, float] = (0.2, 1) 

@configclass
class ExtremeCRLHurdleTerrainCfg(ExtremeCRLRoughTerrainCfg):
    function = crl_terrains.crl_hurdle_terrain
    stone_len: str = '0.1 + 0.3 * difficulty'
    hurdle_height_range: str = '0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty'

@configclass
class ExtremeCRLStepTerrainCfg(ExtremeCRLRoughTerrainCfg):
    function = crl_terrains.crl_step_terrain
    step_height: str = '0.1 + 0.35*difficulty'

@configclass
class ExtremeCRLTerrainCfg(ExtremeCRLRoughTerrainCfg):
    function = crl_terrains.crl_terrain
    pit_depth: tuple[float, float] = (0.2, 1)
    stone_width: float = 1.0
    last_stone_len: float =1.6
    x_range: str = '-0.1, 0.1+0.3*difficulty'
    y_range: str = '0.2, 0.3+0.1*difficulty'
    stone_len: str = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty'
    incline_height: str = '0.25*difficulty'
    last_incline_height: str = 'incline_height + 0.1 - 0.1*difficulty'

@configclass
class ExtremeCRLDemoTerrainCfg(ExtremeCRLRoughTerrainCfg):
    function = crl_terrains.crl_demo_terrain
