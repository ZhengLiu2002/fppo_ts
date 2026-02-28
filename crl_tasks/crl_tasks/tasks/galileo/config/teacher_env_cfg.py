"""Teacher environment configuration for Galileo CRL tasks."""

from __future__ import annotations

import copy

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from crl_isaaclab.envs import CRLManagerBasedRLEnvCfg

from .defaults import (
    GALILEO_ROUGH_TERRAIN_CFG,
    VIEWER_CFG,
    GalileoBaseSceneCfg,
    GalileoDefaults,
    build_galileo_robot_cfg,
)
from .mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    TeacherCostsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)


@configclass
class GalileoCRLSceneCfg(GalileoBaseSceneCfg):
    robot: ArticulationCfg = build_galileo_robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        # 指向 /World 以捕获地面与静态/动态几何
        mesh_prim_paths=["/World"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        track_air_time=True,
        debug_vis=False,
        force_threshold=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # Ensure robot config uses Galileo (override base scene defaults).
        self.robot = build_galileo_robot_cfg()
        # Rough terrain mix for omni-directional locomotion.
        self.terrain.terrain_generator = copy.deepcopy(GALILEO_ROUGH_TERRAIN_CFG)
        self.terrain.terrain_generator.size = GalileoDefaults.terrain.size
        self.terrain.terrain_generator.border_width = GalileoDefaults.terrain.border_width
        self.terrain.terrain_generator.num_rows = GalileoDefaults.terrain.num_rows
        self.terrain.terrain_generator.num_cols = GalileoDefaults.terrain.num_cols
        self.terrain.terrain_generator.horizontal_scale = GalileoDefaults.terrain.horizontal_scale
        self.terrain.terrain_generator.vertical_scale = GalileoDefaults.terrain.vertical_scale
        self.terrain.terrain_generator.slope_threshold = GalileoDefaults.terrain.slope_threshold
        self.terrain.terrain_generator.curriculum = GalileoDefaults.terrain.curriculum
        self.terrain.terrain_generator.difficulty_range = GalileoDefaults.terrain.difficulty_range


@configclass
class GalileoTeacherCRLEnvCfg(CRLManagerBasedRLEnvCfg):
    scene: GalileoCRLSceneCfg = GalileoCRLSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    costs: TeacherCostsCfg = TeacherCostsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crl_events = None
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = GalileoDefaults.general.decimation
        self.episode_length_s = GalileoDefaults.general.episode_length_s
        self.sim.dt = GalileoDefaults.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024

        self.scene.terrain.terrain_generator.curriculum = GalileoDefaults.terrain.curriculum
        self.events.random_camera_position = None
        self.events.push_robot_vel.interval_range_s = (8.0, 8.0)
        # sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        # ensure mass/com events target base_link
        self.events.randomize_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_base_com.params["asset_cfg"].body_names = "base_link"
        self.events.push_robot_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoTeacherCRLEnvCfg_PLAY(GalileoTeacherCRLEnvCfg):
    viewer = VIEWER_CFG

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.debug_vis = True
