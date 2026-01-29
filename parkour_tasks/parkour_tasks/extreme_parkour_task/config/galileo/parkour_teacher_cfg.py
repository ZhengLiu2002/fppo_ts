import torch

import copy
import os
from pathlib import Path

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
import isaaclab.sim as sim_utils

from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER
from .parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)

try:
    from galileo_parkour.assets.galileo import GALILEO_CFG
except ImportError:
    # 当外部 galileo_parkour 扩展未安装时，回落到本地 USD，方便单机调试
    GALILEO_CFG = None

# USD 路径：优先环境变量 GALILEO_USD_PATH，其次仓库相对默认值
_DEFAULT_GALILEO_USD = Path(__file__).resolve().parents[5] / "source/extensions/galileo_parkour/galileo_parkour/assets/usd/robot/galileo_v2d3.usd"
GALILEO_USD_PATH = os.environ.get("GALILEO_USD_PATH", str(_DEFAULT_GALILEO_USD))
# 可用固定栏杆高度（厘米）与颜色，用于训练/竞赛布局
# 栏杆高度档位（米），全局统一供课程、固定赛道等模式复用
HURDLE_HEIGHTS_CM = (5, 10, 20, 30, 40, 50)
HURDLE_HEIGHTS_M = tuple(h / 100.0 for h in HURDLE_HEIGHTS_CM)
HURDLE_BAR_LENGTH = 1.6
HURDLE_BAR_THICKNESS = 0.07
HURDLE_BAR_DEPTH = 0.06
HURDLE_BASE_THICKNESS = 0.04
HURDLE_BASE_DEPTH = 0.18
HURDLE_POST_RADIUS = 0.05
HURDLE_POST_Y_OFFSET = HURDLE_BAR_LENGTH * 0.5 - HURDLE_POST_RADIUS
HURDLE_MODE_NONE = -1
HURDLE_MODE_JUMP = 0
HURDLE_MODE_CRAWL = 1


def _hurdle_asset_name(height_cm: int, component: str) -> str:
    return f"hurdle_{height_cm}cm_{component}"  # 组件命名统一，便于场景索引


def _hurdle_color(height_cm: int) -> tuple[float, float, float]:
    palette = {
        20: (0.20, 0.52, 0.85),
        30: (0.27, 0.68, 0.52),
        40: (0.86, 0.62, 0.26),
        50: (0.83, 0.32, 0.32),
        10: (0.55, 0.45, 0.80),
        5: (0.35, 0.70, 0.88),
    }
    return palette.get(height_cm, (0.6, 0.6, 0.6))


def _galileo_robot_cfg():
    """构造 Galileo 机器人关节/资产配置，若安装了扩展则复用官方配置。"""
    if GALILEO_CFG is not None:
        return GALILEO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GALILEO_USD_PATH,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
        ),
        actuators={},
    )


def place_galileo_hurdles(
    env,
    env_ids,
    spacing: float = 2.0,
    start: float = 2.0,
    layout: str = "auto",
    jump_to_mix_level: int = 6,
    mix_refresh_prob: float = 0.1,
    warmup_levels: int = 1,
    early_crawl_prob: float = 0.15,
):
    """单跑道栏杆摆放：所有 env 沿跑道依次放置 4 根 20/30/40/50cm 栏杆。

    - 课程模式（layout="auto" 且启用 curriculum）时，随着阶段提升逐渐解锁更多栏杆，
      并通过线性插值让高度从热身值平滑过渡到目标高度（20/30/40/50cm）。
    - 固定/竞赛模式下直接输出完整序列，避免每列策略固化。
    - 栏杆模式标记依据实际高度（<=0.35m 视作跳跃，>0.35m 视作钻爬），奖励统计与日志均可复用。
    """
    terrain = env.scene.terrain
    terrain_levels = terrain.terrain_levels[env_ids]
    env_origins = terrain.env_origins[env_ids]
    curriculum_on = getattr(terrain.cfg.terrain_generator, "curriculum", True)
    num_slots = 4
    target_seq = torch.tensor([0.20, 0.30, 0.40, 0.50], device=env.device)
    warmup_seq = torch.tensor([0.12, 0.18, 0.26, 0.34], device=env.device)
    unlock_interval = max(int(warmup_levels), 1)
    jump_threshold = 0.35

    if layout != "auto" or not curriculum_on:
        num_visible = torch.full_like(terrain_levels, num_slots)
        target_h = target_seq.repeat(len(env_ids), 1)
    else:
        stage_tensor = getattr(env, "curriculum_stage", None)
        if stage_tensor is not None:
            stage_used = torch.clamp(stage_tensor[env_ids], 0, 50)
        else:
            stage_used = torch.clamp((terrain_levels // 2), 0, 50)
        num_visible = torch.clamp(stage_used // unlock_interval + 1, max=num_slots)
        # 热身到正式阶段的线性插值：stage 达 jump_to_mix_level 后输出最终高度
        denom = max(float(jump_to_mix_level), 1.0)
        progress = torch.clamp(stage_used.float() / denom, 0.0, 1.0).unsqueeze(1)
        target_h = torch.lerp(warmup_seq, target_seq, progress)
        full_mask = stage_used >= jump_to_mix_level
        if torch.any(full_mask):
            repeat_n = int(full_mask.sum().item())
            target_h[full_mask] = target_seq.repeat(repeat_n, 1)

        # Ensure a small fraction of envs see high (crawl) hurdles even at low stages.
        # This prevents the crawl expert from staying "empty" while keeping jump-first curriculum.
        if early_crawl_prob > 0.0:
            low_stage_mask = stage_used < max(int(jump_to_mix_level), 1)
            sample_mask = low_stage_mask & (torch.rand_like(stage_used.float()) < early_crawl_prob)
            if torch.any(sample_mask):
                high_choices = torch.tensor([0.40, 0.50], device=env.device)
                high_idx = torch.randint(0, high_choices.numel(), (int(sample_mask.sum().item()),), device=env.device)
                target_h[sample_mask, 0] = high_choices[high_idx]

    target_x = env_origins[:, 0].unsqueeze(1) + start + spacing * torch.arange(num_slots, device=env.device)
    target_y = env_origins[:, 1].unsqueeze(1)

    avail_heights = torch.tensor(HURDLE_HEIGHTS_M, device=env.device)
    # 缓存每个 env 的实际栏杆高度/模式，供特权观测/奖励使用
    if not hasattr(env.scene, "hurdle_heights"):
        env.scene.hurdle_heights = torch.full((env.num_envs, 4), -1.0, device=env.device)
    if not hasattr(env.scene, "hurdle_modes"):
        env.scene.hurdle_modes = torch.full((env.num_envs, 4), HURDLE_MODE_NONE, device=env.device, dtype=torch.long)
    env.scene.hurdle_heights[env_ids] = -1.0  # 默认无栏杆
    env.scene.hurdle_modes[env_ids] = HURDLE_MODE_NONE

    # 先隐藏全部组件，避免上一次重置的残留
    for height_cm in HURDLE_HEIGHTS_CM:
        for component in ("base", "bar", "post_left", "post_right"):
            asset_name = _hurdle_asset_name(height_cm, component)
            try:
                asset = env.scene[asset_name]
            except KeyError:
                continue
            hide_state = asset.data.default_root_state[env_ids].clone()
            hide_state[:, 0:3] = -1000.0
            asset.write_root_pose_to_sim(hide_state[:, :7], env_ids)
            asset.write_root_velocity_to_sim(torch.zeros_like(hide_state[:, 7:]), env_ids)

    # 按栏杆槽位逐一放置，并为每个 env 选择最接近的预制高度
    for idx in range(num_slots):
        active_mask = num_visible > idx
        if not active_mask.any():
            continue
        desired_h = target_h[:, idx]
        # 选择最近高度的资产
        diff = torch.abs(desired_h.unsqueeze(1) - avail_heights.unsqueeze(0))
        chosen_idx = torch.argmin(diff, dim=1)

        for asset_choice, height_cm in enumerate(HURDLE_HEIGHTS_CM):
            asset_h = avail_heights[asset_choice]
            env_sel = active_mask & (chosen_idx == asset_choice)
            active_ids = env_ids[env_sel]
            if len(active_ids) == 0:
                continue

            env.scene.hurdle_heights[active_ids, idx] = asset_h
            mode_val = HURDLE_MODE_CRAWL if float(asset_h) > jump_threshold else HURDLE_MODE_JUMP
            env.scene.hurdle_modes[active_ids, idx] = torch.full(
                (active_ids.numel(),), mode_val, device=env.device, dtype=torch.long
            )
            _place_static_hurdle(
                env=env,
                height_cm=height_cm,
                bar_height=asset_h,
                x_positions=target_x[env_sel, idx],
                y_positions=target_y[env_sel, 0],
                ground_heights=env_origins[env_sel, 2],
                active_ids=active_ids,
            )


def _place_static_hurdle(env, height_cm, bar_height, x_positions, y_positions, ground_heights, active_ids):
    """在场景中放置静态原语栏杆（底座+两根立柱+横杆），使用运动学体避免物理漂移。"""
    base_z = ground_heights + HURDLE_BASE_THICKNESS * 0.5
    bar_z = ground_heights + HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS * 0.5
    post_height = HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS
    post_z = ground_heights + post_height * 0.5
    zeros_vel = torch.zeros((active_ids.numel(), 6), device=env.device)

    def _set_pose(asset_name: str, pos_x: torch.Tensor, pos_y: torch.Tensor, pos_z: torch.Tensor):
        asset = env.scene[asset_name]
        root_state = asset.data.default_root_state[active_ids].clone()
        root_state[:, 0] = pos_x
        root_state[:, 1] = pos_y
        root_state[:, 2] = pos_z
        asset.write_root_pose_to_sim(root_state[:, :7], active_ids)
        asset.write_root_velocity_to_sim(zeros_vel, active_ids)

    _set_pose(_hurdle_asset_name(height_cm, "base"), x_positions, y_positions, base_z)
    _set_pose(_hurdle_asset_name(height_cm, "bar"), x_positions, y_positions, bar_z)
    _set_pose(_hurdle_asset_name(height_cm, "post_left"), x_positions, y_positions - HURDLE_POST_Y_OFFSET, post_z)
    _set_pose(_hurdle_asset_name(height_cm, "post_right"), x_positions, y_positions + HURDLE_POST_Y_OFFSET, post_z)


@configclass
class GalileoParkourSceneCfg(ParkourDefaultSceneCfg):
    robot: ArticulationCfg = _galileo_robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        # 指向 /World 以捕获地面与栏杆等所有静态/动态几何
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
        # Ensure robot config uses Galileo (override go2 defaults from base scene).
        self.robot = _galileo_robot_cfg()
        # Flat terrain; curriculum drives hurdle count/height progression.
        self.terrain.terrain_generator = copy.deepcopy(EXTREME_PARKOUR_TERRAINS_CFG)
        # Extend lane length so entry + 4 hurdles + final goal markers all stay inside bounds
        self.terrain.terrain_generator.size = (20.0, self.terrain.terrain_generator.size[1])
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 4
        self.terrain.terrain_generator.horizontal_scale = 0.1
        # Waypoints follow the hurdle layout: entry + 4 hurdles + final target
        self.terrain.terrain_generator.num_goals = 6
        self.terrain.terrain_generator.curriculum = True
        self.terrain.terrain_generator.random_difficulty = False
        self.terrain.terrain_generator.difficulty_range = (0.0, 0.15)
        # 阶段制课程：使用完整行数供阶段映射，初始从 0 开始，确保“跳/钻”列的分布稳定
        self.terrain.max_init_terrain_level = 0
        self.terrain.max_terrain_level = 10
        for name, sub in self.terrain.terrain_generator.sub_terrains.items():
            sub.use_simplified = True
            sub.proportion = 1.0 if name == "parkour_flat" else 0.0
            sub.apply_roughness = False

        # Hurdle assets (static primitives: base + posts + bar). 统一预生成 6 个高度档的 4 个组件
        base_size = (HURDLE_BASE_DEPTH, HURDLE_BAR_LENGTH + 0.2, HURDLE_BASE_THICKNESS)
        bar_size = (HURDLE_BAR_DEPTH, HURDLE_BAR_LENGTH, HURDLE_BAR_THICKNESS)
        for h_cm, bar_height in zip(HURDLE_HEIGHTS_CM, HURDLE_HEIGHTS_M):
            color = _hurdle_color(h_cm)
            base_spawn = sim_utils.CuboidCfg(
                size=base_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            bar_spawn = sim_utils.CuboidCfg(
                size=bar_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            post_spawn = sim_utils.CylinderCfg(
                radius=HURDLE_POST_RADIUS,
                height=HURDLE_BASE_THICKNESS + bar_height + HURDLE_BAR_THICKNESS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            component_cfgs = {
                "base": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'base')}",
                    spawn=base_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "bar": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'bar')}",
                    spawn=bar_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "post_left": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'post_left')}",
                    spawn=post_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
                "post_right": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hurdle_asset_name(h_cm, 'post_right')}",
                    spawn=post_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-1000.0, -1000.0, -1000.0),
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                ),
            }
            for comp, cfg in component_cfgs.items():
                setattr(self, _hurdle_asset_name(h_cm, comp), cfg)


@configclass
class GalileoTeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: GalileoParkourSceneCfg = GalileoParkourSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
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

        # 保持课程开启，用栏杆数量/高度逐步提难度；混合课程在 reset 时布置
        self.scene.terrain.terrain_generator.curriculum = True
        self.events.random_camera_position = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        # place hurdles on reset
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
        # sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        # ensure mass/com events target base_link
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = "base_link"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"


@configclass
class GalileoTeacherParkourEnvCfg_PLAY(GalileoTeacherParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        # Play 时默认展示固定比赛布局（20/30/40/50）并开启可视化箭头/航向点
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.events.place_hurdles.params["start"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["spacing"] = 2.0  # type: ignore[attr-defined]
        self.events.place_hurdles.params["layout"] = "competition"  # type: ignore[attr-defined] fixed/competition
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
