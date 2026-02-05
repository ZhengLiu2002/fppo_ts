import math
import os
import torch
from pathlib import Path
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import ViewerCfg
from crl_isaaclab.actuators.crl_actuator_cfg import CRLDCMotorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

try:
    from galileo_parkour.assets.galileo import GALILEO_CFG
except ImportError:
    GALILEO_CFG = None

# 路径计算：config_summary.py 位于 crl_tasks/crl_tasks/crl_task/config/galileo/
# parents[0]=galileo, [1]=config, [2]=crl_task, [3]=crl_tasks, [4]=crl_tasks, [5]=项目根目录
_DEFAULT_GALILEO_USD = Path(__file__).resolve().parents[5] / "source/extensions/galileo_parkour/galileo_parkour/assets/usd/robot/galileo_v2d3.usd"
GALILEO_USD_PATH = os.environ.get("GALILEO_USD_PATH", str(_DEFAULT_GALILEO_USD))


def _galileo_robot_cfg() -> ArticulationCfg:
    """Galileo 机器人配置函数"""
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
                "FL_hip_joint": -0.05,
                "FL_thigh_joint": 0.795,
                "FL_calf_joint": -1.61,
                "FR_hip_joint": 0.05,
                "FR_thigh_joint": 0.795,
                "FR_calf_joint": -1.61,
                "RL_hip_joint": -0.05,
                "RL_thigh_joint": 0.795,
                "RL_calf_joint": -1.61,
                "RR_hip_joint": 0.05,
                "RR_thigh_joint": 0.795,
                "RR_calf_joint": -1.61,
            },
        ),
        actuators={},
    )


def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    """从欧拉角转换为四元数（ROS 约定）"""
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
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())


@configclass
class CRLDefaultSceneCfg(InteractiveSceneCfg):
    """CRL 默认场景配置"""
    robot: ArticulationCfg = _galileo_robot_cfg()
    
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
        self.robot.actuators['base_legs'] = CRLDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={
                '.*_hip_joint':120.0,
                '.*_thigh_joint':120.0,
                '.*_calf_joint':120.0,
            },
            saturation_effort={
                '.*_hip_joint':120.0,
                '.*_thigh_joint':120.0,
                '.*_calf_joint':120.0,
            },
            velocity_limit={
                '.*_hip_joint':16.0,
                '.*_thigh_joint':16.0,
                '.*_calf_joint':16.0,
            },
            stiffness=70.0,
            damping=2.0,
            friction=0.05,
        )


VIEWER = ViewerCfg(
    eye=(-0., 2.6, 1.6),
    asset_name = "robot",
    origin_type = 'asset_root',
)


# ========== 地形配置 ==========
# Rough terrain mix used for Galileo FPPO tasks.
GALILEO_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.30,
        ),
    },
)


class ConfigSummary:
    class general:
        decimation = 4
        episode_length_s = 20.0
        render_interval = 4

    class curriculum:
        episodes_per_level = 700

    class sim:
        dt = 0.0025

    class env:
        num_envs = 4096

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
        min_abs_lin_vel_x: float = 0.3
        min_abs_lin_vel_y: float = 0.1
        heading_control_stiffness = 0.8
        
        # 默认配置（用于 CommandsCfg 的初始化，不在地形特定配置中）
        class default:
            heading_command_prob: float = 1.0
            standing_command_prob: float = 0.0
            yaw_command_prob: float = 0.0
            lin_vel_x = (0.25, 0.9)
            lin_vel_y = (-0.2, 0.2)
            ang_vel_z = (-0.12, 0.12)
            heading = (-math.pi / 3, math.pi / 3)
            start_curriculum_lin_x = (0.15, 0.35)
            start_curriculum_ang_z = (-0.08, 0.08)
            max_curriculum_lin_x = (0.65, 0.95)
            max_curriculum_ang_z = (-0.25, 0.25)
        
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

    class reward:
        class track_lin_vel_xy_exp:
            weight = 3.0
            std = math.sqrt(0.25)
            command_name = "base_velocity"
            min_command_speed = 0.2

        class track_ang_vel_z_exp:
            weight = 1.5
            std = math.sqrt(0.25)
            command_name = "base_velocity"
            min_command_speed = 0.2

        class lin_vel_z_l2:
            weight = -0.1

        class ang_vel_xy_l2:
            weight = -0.05

        class flat_orientation_l2:
            weight = -0.1

        class base_height_l2_fix:
            weight = -1.0
            target_height = 0.426

        class joint_torques_l2:
            weight = -1.0e-06

        class joint_vel_l2:
            weight = -2.0e-5

        class joint_power_distribution:
            weight = -1.0e-6

        class joint_acc_l2:
            weight = -5.0e-9

        class dof_error_l2:
            weight = -0.05

        class hip_pos_l2:
            weight = -0.05

        class action_rate_l2:
            weight = -0.001

        class action_smoothness_l2:
            weight = -0.001

        class feet_air_time:
            weight = 1.0
            threshold = 0.35
            command_name = "base_velocity"

        class feet_slide:
            weight = -0.1

        class load_sharing:
            weight = 0.0

        class undesired_contacts:
            weight = -0.1
            threshold = 0.5

# ------------------------------------------

    class cost:
        class prob_joint_pos:
            weight = 1.0
            margin = -0.05
            limit = 1.0

        class prob_joint_vel:
            weight = 1.0
            velocity_limit = 15.0
            limit = 1.0

        class prob_joint_torque:
            weight = 1.0
            torque_limit = 120.0
            limit = 1.0

        class prob_body_contact:
            weight = 0.0
            contact_force_threshold = 1.0
            limit = 1.0

        class prob_com_frame:
            weight = 0.0
            height_range = (0.2, 0.6)
            max_angle_rad = 1.0
            limit = 5.0

        class prob_gait_pattern:
            weight = 0.0
            gait_frequency = 1.5
            min_frequency = None
            max_frequency = None
            max_command_speed = None
            frequency_scale = 0.0
            min_command_speed = None
            min_base_speed = None
            stance_ratio = 0.5
            phase_offsets = [0.0, 0.5, 0.5, 0.0]
            contact_force_threshold = 1.0
            limit = 5.0

        class orthogonal_velocity:
            weight = 0.0
            limit = 3.0

        class contact_velocity:
            weight = 0.0
            contact_force_threshold = 1.0
            limit = 0.8

        class foot_clearance:
            weight = 0.
            min_height = None
            limit = 0.08
            min_command_speed = None
            min_base_speed = None

        class foot_height_limit:
            weight = 0.
            limit = 0.4

        class symmetric:
            weight = 0.0
            limit = 5.0
            command_name = "base_velocity"
            min_command_speed = None
            min_base_speed = None
            joint_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

        class base_contact_force:
            weight = 0.0
            contact_force_threshold = 1.0
            limit = 1.0

        total_limit = 11.0
        aggregate_limit = 1.3

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
                "x": (-1.0, 1.0),  # m/s
                "y": (-1.0, 1.0),  # m/s
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            interval_range_s = (8.0, 8.0)
            is_global_time = True
        
        class push_robot_torque:
            force_range = (0.0, 0.0)
            torque_range = (-3.0, 3.0)  # N·m
            interval_range_s = (8.0, 8.0)
            is_global_time = True


    class algorithm:
        """算法选择与超参配置（集中在 ConfigSummary 中便于调参）。

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

        # 与 CLI `--algo` 对齐：{"fppo","ppo","ppo_lagrange","cpo","pcpo","focpo","distillation"}
        name: str = "fppo"

        # CLI/代码中的 class_name 映射（最终写入 agent_cfg.algorithm.class_name）
        class_name_map = {
            "fppo": "FPPO",
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
            clip_param=0.08,
            desired_kl=0.008,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
            # CMDP common (if used)
            cost_limit=1.3,
        )

        # 各算法差异化字段（只写需要的即可）
        per_algo = {
            # FPPO
            "fppo": dict(
                cost_value_loss_coef=1.0,
                step_size=3.0e-4,
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
            entropy_coef=0.001,  # 进一步降低探索性
        )
        student_override = dict(
            entropy_coef=0.006,  # 进一步降低探索性
            dagger_update_freq=1,  # Student stage-2: always use history latent (DAgger)
        )

    
