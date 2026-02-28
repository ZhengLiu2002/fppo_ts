"""Shared defaults for Galileo CRL tasks."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from crl_isaaclab.actuators.crl_actuator_cfg import CRLDCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

try:
    from galileo_parkour.assets.galileo import GALILEO_CFG as _GALILEO_CFG
except ImportError:
    _GALILEO_CFG = None

# File location: crl_tasks/crl_tasks/tasks/galileo/config/defaults.py
# parents[0]=config, [1]=galileo, [2]=tasks, [3]=crl_tasks, [4]=crl_tasks, [5]=repo root
DEFAULT_GALILEO_USD_PATH = (
    Path(__file__).resolve().parents[5]
    / "source/extensions/galileo_parkour/galileo_parkour/assets/usd/robot/galileo_v2d3.usd"
)
GALILEO_USD_PATH = os.environ.get("GALILEO_USD_PATH", str(DEFAULT_GALILEO_USD_PATH))


def _build_galileo_robot_cfg() -> ArticulationCfg:
    """Build the Galileo articulation configuration."""
    if _GALILEO_CFG is not None:
        return _GALILEO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
                "FL_hip_joint": -0.05,
                "FL_thigh_joint": 0.75,
                "FL_calf_joint": -1.5,
                "FR_hip_joint": 0.05,
                "FR_thigh_joint": 0.75,
                "FR_calf_joint": -1.5,
                "RL_hip_joint": -0.05,
                "RL_thigh_joint": 0.75,
                "RL_calf_joint": -1.5,
                "RR_hip_joint": 0.05,
                "RR_thigh_joint": 0.75,
                "RR_calf_joint": -1.5,
            },
        ),
        actuators={},
    )


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    """Convert Euler XYZ to quaternion (ROS convention)."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.0, 1.0, 1.0, -1.0])
    return tuple(convert.numpy().tolist())


@configclass
class GalileoBaseSceneCfg(InteractiveSceneCfg):
    """Default Galileo scene configuration shared across tasks."""

    robot: ArticulationCfg = _build_galileo_robot_cfg()

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type=TerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.actuators["base_legs"] = CRLDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                ".*_hip_joint": 80.0,
                ".*_thigh_joint": 80.0,
                ".*_calf_joint": 80.0,
            },
            saturation_effort={
                ".*_hip_joint": 80.0,
                ".*_thigh_joint": 80.0,
                ".*_calf_joint": 80.0,
            },
            velocity_limit={
                ".*_hip_joint": 16.0,
                ".*_thigh_joint": 16.0,
                ".*_calf_joint": 16.0,
            },
            stiffness=70.0,
            damping=3.5,
            friction=0.05,
        )


VIEWER_CFG = ViewerCfg(
    eye=(-0.0, 2.6, 1.6),
    asset_name="robot",
    origin_type="asset_root",
)


# ========== 地形配置 ==========
# Rough terrain mix used for Galileo FPPO tasks.
GALILEO_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.20,
        #     step_height_range=(0.05, 0.2),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.20,
        #     step_height_range=(0.05, 0.2),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.15,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.15,
        #     slope_range=(0.0, 0.4),
        #     platform_width=2.0,
        #     border_width=0.25,
        # ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)


class GalileoDefaults:
    class general:
        decimation = 8
        episode_length_s = 20.0
        render_interval = 4

    class curriculum:
        # Curriculum is gated by env.common_step_counter (per-env steps). With large num_envs,
        # 700 episodes/level advances very slowly; reduce for visible progression.
        episodes_per_level = 20

    class sim:
        dt = 0.0025

    class env:
        num_envs = 4096

    class obs:
        """Actor/Critic 观测维度与布局（训练配置从这里读取）。"""

        # 通用环境/动作维度
        num_envs = 4096
        num_actions = 12

        # Teacher / Student 的 actor-critic 观测维度（当前默认一致，后续可按需分化）
        class teacher:
            # Actor obs:
            # base_lin_vel*3 + base_ang_vel*3 + projected_gravity*3 + joint_pos*12
            # + joint_vel*12 + actions*12 + commands*3
            # + height_scan*132 + (base_com*3 + base_mass*1 + ground_friction*1)
            actor_num_prop = 48
            actor_num_scan = 132
            actor_num_priv_explicit = 5
            actor_num_priv_latent = 0
            actor_num_hist = 0
            num_actor_obs = (
                actor_num_prop
                + actor_num_scan
                + actor_num_priv_explicit
                + actor_num_priv_latent
                + actor_num_prop * actor_num_hist
            )

            # Critic obs:
            # base_lin_vel*3 + base_ang_vel*3 + projected_gravity*3 + joint_pos*12 + joint_vel*12 + actions*12 + commands*3
            # + height_scan*132 + (base_com*3 + base_mass*1 + ground_friction*1)
            critic_num_prop = 48
            critic_num_scan = 132
            critic_num_priv_explicit = 5
            critic_num_priv_latent = 0
            critic_num_hist = 0
            num_critic_obs = (
                critic_num_prop
                + critic_num_scan
                + critic_num_priv_explicit
                + critic_num_priv_latent
                + critic_num_prop * critic_num_hist
            )

        class student:
            # Actor obs (同 teacher):
            # base_ang_vel*3 + projected_gravity*3 + joint_pos*12 + joint_vel*12 + actions*12 + commands*3
            actor_num_prop = 45
            actor_num_scan = 0
            actor_num_priv_explicit = 0
            actor_num_priv_latent = 0
            actor_num_hist = 0
            num_actor_obs = (
                actor_num_prop
                + actor_num_scan
                + actor_num_priv_explicit
                + actor_num_priv_latent
                + actor_num_prop * actor_num_hist
            )

            # Critic obs (同 teacher):
            # base_lin_vel*3 + base_ang_vel*3 + projected_gravity*3 + joint_pos*12 + joint_vel*12 + actions*12 + commands*3
            # + height_scan*132 + (base_com*3 + base_mass*1 + ground_friction*1)
            critic_num_prop = 48
            critic_num_scan = 132
            critic_num_priv_explicit = 5
            critic_num_priv_latent = 0
            critic_num_hist = 0
            num_critic_obs = (
                critic_num_prop
                + critic_num_scan
                + critic_num_priv_explicit
                + critic_num_priv_latent
                + critic_num_prop * critic_num_hist
            )

        # 观测历史相关（如需）
        obs_history_length = 5
        action_history_length = 3
        clip_actions = 100.0
        clip_obs = 100.0

    class terrain:
        size = (8.0, 8.0)
        border_width = 60.0
        num_rows = 10
        num_cols = 20
        horizontal_scale = 0.1
        vertical_scale = 0.005
        slope_threshold = 0.75
        curriculum = True
        random_difficulty = False
        difficulty_range = (0.0, 1.0)

    class command:
        lin_x_level: float = 0.08
        max_lin_x_level: float = 1.0
        ang_z_level: float = 0.04
        max_ang_z_level: float = 1.0
        min_abs_lin_vel_x: float = 0.1
        min_abs_lin_vel_y: float = 0.0
        heading_control_stiffness = 0.8

        # 默认配置（用于 CommandsCfg 的初始化，不在地形特定配置中）
        class default:
            heading_command_prob: float = 0.2
            standing_command_prob: float = 0.0
            yaw_command_prob: float = 0.0
            lin_vel_x = (0.1, 0.4)
            lin_vel_y = (-0.1, 0.1)
            ang_vel_z = (-0.08, 0.08)
            heading = (-math.pi / 8, math.pi / 8)
            start_curriculum_lin_x = (0.1, 0.25)
            start_curriculum_ang_z = (-0.05, 0.05)
            max_curriculum_lin_x = (0.35, 0.55)
            max_curriculum_ang_z = (-0.12, 0.12)

        ranges = {
            "pyramid_stairs": dict(
                lin_vel_x=(0.2, 0.8),
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.7, 0.9),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "pyramid_stairs_inv": dict(
                lin_vel_x=(0.15, 0.75),
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.12, 0.25),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.6, 0.85),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "boxes": dict(
                lin_vel_x=(0.2, 0.8),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.7, 0.9),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "random_rough": dict(
                lin_vel_x=(0.2, 0.85),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.7, 0.95),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "hf_pyramid_slope": dict(
                lin_vel_x=(0.2, 0.8),
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.7, 0.9),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "hf_pyramid_slope_inv": dict(
                lin_vel_x=(0.2, 0.8),
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.12, 0.12),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.15, 0.3),
                start_curriculum_ang_z=(-0.08, 0.08),
                max_curriculum_lin_x=(0.7, 0.9),
                max_curriculum_ang_z=(-0.25, 0.25),
            ),
            "plane_run": dict(
                lin_vel_x=(0.3, 1.1),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.18, 0.18),
                heading=(-math.pi / 3, math.pi / 3),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.2, 0.5),
                start_curriculum_ang_z=(-0.1, 0.1),
                max_curriculum_lin_x=(0.8, 1.2),
                max_curriculum_ang_z=(-0.3, 0.3),
            ),
        }

        resampling_time_range = (6.0, 6.0)
        clips = dict(lin_vel_clip=0.2, ang_vel_clip=0.3)

    class priv_obs_norm:
        """Normalization ranges for privileged observations."""

        base_mass_delta_range = (-3.0, 5.0)
        base_com_range = {
            "x": (-0.05, 0.05),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.05),
        }
        ground_friction_range = (0.25, 1.2)

    class event:
        """域随机化参数配置"""

        # ========== 一、机身/动力学（reset 时） ==========
        class randomize_base_mass:
            mass_distribution_params = (-3.0, 5.0)  # kg（加性）
            operation = "add"

        class randomize_base_com:
            com_range = {
                "x": (-0.05, 0.05),  # m
                "y": (-0.03, 0.03),  # m
                "z": (-0.03, 0.05),  # m
            }

        class physics_material:
            friction_range = (0.25, 1.2)  # 静摩擦和动摩擦使用相同范围
            restitution_range = (0.0, 1.0)  # 恢复系数
            num_buckets = 64

        # ========== 二、重置位姿/速度（reset 时） ==========
        class reset_base_pose:
            pose_range = {
                "x": (-0.5, 0.5),  # m
                "y": (-0.5, 0.5),  # m
                "yaw": (-math.pi, math.pi),  # rad
            }
            velocity_range = {
                "x": (-0.5, 0.5),  # m/s
                "y": (-0.5, 0.5),  # m/s
                "z": (-0.5, 0.5),  # m/s
                "roll": (0.0, 0.0),  # rad/s（角速度不随机）
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }

        # ========== 三、腿部关节（reset 时） ==========
        class reset_leg_joints:
            position_range = (-0.8, 0.8)  # 标幺/偏移
            velocity_range = (0.0, 0.0)

        # ========== 四、执行器增益（reset 时） ==========
        class randomize_actuator_kp_kd_gains:
            stiffness_distribution_params = (0.8, 1.2)  # Kp 乘性
            damping_distribution_params = (0.8, 1.2)  # Kd 乘性
            operation = "scale"
            distribution = "uniform"

        # ========== 五、外力扰动（interval，训练中周期施加） ==========
        class push_robot_vel:
            velocity_range = {
                "x": (-0.3, 0.3),  # m/s
                "y": (-0.3, 0.3),  # m/s
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            interval_range_s = (8.0, 8.0)
            is_global_time = True

        class push_robot_torque:
            force_range = (0.0, 0.0)
            torque_range = (-1.0, 1.0)  # N·m
            interval_range_s = (8.0, 8.0)
            is_global_time = True

    class algorithm:
        """算法选择与超参配置（集中在 GalileoDefaults 中便于调参）。

        设计原则（解决“不同算法参数种类/数量不同”的问题）：
        - **name**: 选择当前使用的算法（与 CLI `--algo` 的取值一致：fppo/ppo/cpo/...）。
        - **base**: 所有算法共享/通用的超参（或你希望大多数算法都用的默认值）。
        - **per_algo**: 针对不同算法的“可选字段”字典，只写该算法需要/关心的字段即可。
        - **teacher_override / student_override**: Teacher/Student 的差异化覆写（可选）。
        - 应用时按顺序合并：base -> per_algo[name] -> teacher/student_override。

        说明：
        - 我们当前的 `CRLRslRlPpoAlgorithmCfg` 是一个“超集”配置（包含 FPPO/CMDP 扩展字段），
          对于 PPO/CPO/PCPO 等算法，未用到的字段会被忽略（由算法实现决定）。
        - 若你希望“严格模式”，可以在应用时对未知字段做过滤/报错（后续可加）。
        """

        # 与 CLI `--algo` 对齐：{"fppo","np3o","ppo","ppo_lagrange","cpo","pcpo","focpo","distillation"}
        name: str = "fppo"

        # CLI/代码中的 class_name 映射（最终写入 agent_cfg.algorithm.class_name）
        class_name_map = {
            "fppo": "FPPO",
            "np3o": "NP3O",
            "ppo": "PPO",
            "ppo_lagrange": "PPOLagrange",
            "cpo": "CPO",
            "pcpo": "PCPO",
            "focpo": "FOCPO",
            "distillation": "Distillation",
        }

        # 共享默认值（两边都通用）
        base = dict(
            # PPO common
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            desired_kl=0.004,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
            # CMDP common
            cost_limit=1.3,
            # NP3O-style shared shaping
            cost_viol_loss_coef=0.1,
            k_value=0.2,
            k_growth=1.0003,
            k_max=1.0,
        )

        # 各算法差异化字段（只写需要的即可）
        per_algo = {
            # FPPO
            "fppo": dict(
                cost_value_loss_coef=1.0,
                step_size=1.5e-4,
                cost_gamma=None,
                cost_lam=None,
                delta_safe=0.01,
                backtrack_coeff=0.5,
                max_backtracks=10,
                projection_eps=1e-8,
                normalize_cost_advantage=False,
                constraint_normalization=True,
                constraint_norm_beta=0.99,
                constraint_norm_min_scale=1e-3,
                constraint_norm_max_scale=10.0,
                constraint_norm_clip=5.0,
                constraint_proxy_delta=0.1,
                constraint_agg_tau=0.5,
                constraint_scale_by_gamma=True,
                use_preconditioner=True,
                preconditioner_beta=0.999,
                preconditioner_eps=1e-8,
                feasible_first=True,
                feasible_first_coef=1.0,
                dagger_update_freq=20,
                priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0],
            ),
            # NP3O
            "np3o": dict(
                cost_value_loss_coef=1.0,
                cost_gamma=None,
                cost_lam=None,
                normalize_cost_advantage=False,
                cost_viol_loss_coef=1.0,
                k_value=0.05,
                k_growth=1.0004,
                k_max=1.0,
                dagger_update_freq=20,
            ),
            # PPO（示例：如果你切到 PPO，只需要写 PPO 特有/你要覆写的字段）
            "ppo": dict(),
            # 其他算法暂时留空：后续按需在这里补字段即可
            "ppo_lagrange": dict(),
            "cpo": dict(),
            "pcpo": dict(),
            "focpo": dict(),
            "distillation": dict(),
        }

        # Teacher/Student 差异（可选）
        teacher_override = dict(
            entropy_coef=0.005,
        )
        student_override = dict(
            entropy_coef=0.01,
            dagger_update_freq=1,  # Student stage-2: always use history latent (DAgger)
        )


# -----------------------------------------------------------------------------
# Convenience aliases for common naming patterns.
# -----------------------------------------------------------------------------

ConfigSummary = GalileoDefaults
CRLDefaultSceneCfg = GalileoBaseSceneCfg
VIEWER = VIEWER_CFG
GALILEO_ROUGH_TERRAINS_CFG = GALILEO_ROUGH_TERRAIN_CFG

build_galileo_robot_cfg = _build_galileo_robot_cfg
_galileo_robot_cfg = _build_galileo_robot_cfg

quat_from_euler_xyz_tuple = quat_from_euler_xyz

__all__ = [
    "DEFAULT_GALILEO_USD_PATH",
    "GALILEO_USD_PATH",
    "build_galileo_robot_cfg",
    "_galileo_robot_cfg",
    "quat_from_euler_xyz",
    "quat_from_euler_xyz_tuple",
    "GalileoBaseSceneCfg",
    "CRLDefaultSceneCfg",
    "VIEWER_CFG",
    "VIEWER",
    "GALILEO_ROUGH_TERRAIN_CFG",
    "GALILEO_ROUGH_TERRAINS_CFG",
    "GalileoDefaults",
    "ConfigSummary",
]
