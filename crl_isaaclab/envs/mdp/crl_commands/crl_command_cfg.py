from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass

import math
from dataclasses import MISSING
from .uniform_crl_command import UniformCRLCommand


@configclass
class CRLCommandCfg(CommandTermCfg):
    class_type: type = UniformCRLCommand
    asset_name: str = MISSING
    heading_control_stiffness: float = 1.0
    small_commands_to_zero: bool = True
    heading_command: bool = True
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.0
    align_to_crl_goal: bool = False
    # If enabled, command ranges are additionally scaled by terrain levels in
    # UniformCRLCommand._resample_command. Keep disabled when using curriculum
    # terms as the single source of command difficulty.
    terrain_level_range_scaling: bool = False
    # curriculum levels for progressive command ranges
    lin_x_level: float = 0.0
    max_lin_x_level: float = 1.0
    lin_x_level_step: float = 0.1
    ang_z_level: float = 0.0
    max_ang_z_level: float = 1.0
    ang_z_level_step: float = 0.1
    # minimum absolute command magnitudes (optional)
    min_abs_lin_vel_x: float | None = None
    min_abs_lin_vel_y: float | None = None

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = (-0.35, 0.35)
        ang_vel_z: tuple[float, float] = (-0.2, 0.2)
        heading: tuple[float, float] | None = MISSING
        heading_command_prob: float = 1.0
        yaw_command_prob: float = 0.0
        standing_command_prob: float = 0.0
        start_curriculum_lin_x: tuple[float, float] | None = None
        start_curriculum_ang_z: tuple[float, float] | None = None
        max_curriculum_lin_x: tuple[float, float] | None = None
        max_curriculum_ang_z: tuple[float, float] | None = None

    @configclass
    class Clips:
        lin_vel_clip: float = MISSING
        ang_vel_clip: float = MISSING

    ranges: Ranges = MISSING
    clips: Clips = MISSING

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
