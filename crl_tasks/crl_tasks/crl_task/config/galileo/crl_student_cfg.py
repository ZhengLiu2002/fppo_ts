from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from crl_isaaclab.envs import CRLManagerBasedRLEnvCfg
from .config_summary import VIEWER
from .crl_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    StudentCostsCfg,
    StudentObservationsCfg,
    StudentRewardsCfg,
    TerminationsCfg,
)
from .crl_teacher_cfg import GalileoCRLSceneCfg
from .config_summary import ConfigSummary


@configclass
class GalileoStudentSceneCfg(GalileoCRLSceneCfg):
    # Blind student: remove exteroceptive sensors.
    height_scanner = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=2,
        track_air_time=False,
        debug_vis=False,
        force_threshold=1.0,
    )


@configclass
class GalileoStudentCRLEnvCfg(CRLManagerBasedRLEnvCfg):
    scene: GalileoStudentSceneCfg = GalileoStudentSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    costs: StudentCostsCfg = StudentCostsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crl_events = None
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = ConfigSummary.general.decimation
        self.episode_length_s = ConfigSummary.general.episode_length_s
        self.sim.dt = ConfigSummary.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024
        self.scene.terrain.terrain_generator.curriculum = ConfigSummary.terrain.curriculum
        # 蒸馏/学生策略开启关节延迟与历史，模拟实际执行链路的滞后
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
        self.events.random_camera_position = None
        # ensure mass/com events target base_link
        self.events.randomize_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoStudentCRLEnvCfg_EVAL(GalileoStudentCRLEnvCfg):
    viewer = VIEWER
    rewards: StudentRewardsCfg = StudentRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 评估/可视化：减少并行环境数量、放宽命令采样，开启调试可视化
        self.scene.num_envs = 128
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_base_com = None
        self.events.randomize_base_mass = None
        self.events.push_robot_vel.interval_range_s = (6.0, 6.0)


@configclass
class GalileoStudentCRLEnvCfg_PLAY(GalileoStudentCRLEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        # 试玩模式：降低并行数、延长单回合时长，并切换为竞赛固定布局
        self.scene.num_envs = 16
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.push_robot_vel = None
