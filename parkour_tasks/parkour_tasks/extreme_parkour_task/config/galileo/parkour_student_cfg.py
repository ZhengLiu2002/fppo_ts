from isaaclab.utils import configclass

from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.default_cfg import CAMERA_CFG, VIEWER
from .parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    StudentObservationsCfg,
    StudentRewardsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)
from .parkour_teacher_cfg import GalileoParkourSceneCfg


@configclass
class GalileoStudentSceneCfg(GalileoParkourSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None

    def __post_init__(self):
        super().__post_init__()
        # 与教师环境保持同样的行列布局，复用栏杆混合课程与站位逻辑
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 4
        self.terrain.terrain_generator.horizontal_scale = 0.1


@configclass
class GalileoStudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: GalileoStudentSceneCfg = GalileoStudentSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.random_difficulty = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.15)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.max_terrain_level = 10
        # 蒸馏/学生策略开启关节延迟与历史，模拟实际执行链路的滞后
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8
        # place hurdles on reset
        if not hasattr(self.events, "place_hurdles"):
            from parkour_tasks.extreme_parkour_task.config.galileo.parkour_teacher_cfg import place_galileo_hurdles
            from isaaclab.managers import EventTermCfg

            self.events.place_hurdles = EventTermCfg(
                func=place_galileo_hurdles,
                mode="reset",
                params={
                    "spacing": 2.0,
                    "start": 2.0,
                    "layout": "auto",
                    "jump_to_mix_level": 8,
                    "mix_refresh_prob": 0.1,
                    "warmup_levels": 2,
                },
            )
        self.events.place_hurdles.params["spacing"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["start"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["layout"] = "auto"  # type: ignore[attr-defined]
        self.events.place_hurdles.params["jump_to_mix_level"] = 8  # type: ignore[attr-defined]
        self.events.place_hurdles.params["warmup_levels"] = 2  # type: ignore[attr-defined]
        # ensure mass/com events target base_link
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoStudentParkourEnvCfg_EVAL(GalileoStudentParkourEnvCfg):
    viewer = VIEWER
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 评估/可视化：减少并行环境数量、放宽命令采样，开启调试可视化
        self.scene.num_envs = 128
        self.commands.base_velocity.debug_vis = True
        self.scene.depth_camera.params["debug_vis"] = True  # type: ignore[index]
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.scene.terrain.terrain_generator.random_difficulty = True
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)


@configclass
class GalileoStudentParkourEnvCfg_PLAY(GalileoStudentParkourEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        # 试玩模式：降低并行数、延长单回合时长，并切换为竞赛固定布局
        self.scene.num_envs = 16
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.place_hurdles.params["layout"] = "competition"  # type: ignore[attr-defined]
        self.events.push_by_setting_velocity = None
