from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.events import (
    randomize_rigid_body_mass,
    apply_external_force_torque,
    reset_joints_by_scale,
)
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands


@configclass
class CommandsCfg:
    """前进速度指令（针对直线跑道栏杆）

    - 仅在 X 方向采样 0.6~1.2 m/s，保持训练聚焦在跨栏核心动作。
    - heading 控制设置轻微偏移范围，利于生成左右对称的轨迹。
    """

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        heading_control_stiffness=1.0,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.2),
            heading=(-0.2, 0.2),
        ),
        clips=parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.3,
        ),
    )


@configclass
class ParkourEventsCfg:
    # 课程推进/退阶阈值：更高的 promotion_goal_threshold 鼓励稳定越障再升级
    base_parkour = parkours.ParkourEventsCfg(
        asset_name="robot",
        promotion_goal_threshold=0.9,
        demotion_goal_threshold=0.35,
        promotion_distance_ratio=0.8,
        demotion_distance_ratio=0.4,
        distance_progress_cap=12.0,
    )


@configclass
class TeacherObservationsCfg:
    """教师（教师网络）观测定义：只包含状态/激光，不含摄像头。"""

    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class StudentObservationsCfg:
    """学生（蒸馏/学生网络）观测：额外包含深度相机与航向判定。"""
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
                "include_privileged": False,
            },
            clip=(-100, 100),
        )

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": False,
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        deta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={"parkour_name": "base_parkour", "threshold": 0.6},
        )

    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class StudentRewardsCfg:
    """学生奖励（轻量，避免过拟合特权）

    - 主要围绕安全越障（高度引导/跨越/钻爬/模式匹配）与平稳性（扭矩、动作变化）。
    - 部分奖励（reward_alive 等）简化权重，便于蒸馏到摄像头输入的学生策略。
    """
    reward_alive = RewTerm(
        func=rewards.reward_alive,
        weight=.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-6.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_calf", ".*_thigh"])},
    )
    reward_height_guidance = RewTerm(
        func=rewards.reward_height_guidance,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "target_height": 0.4,
            "speed_gate": 0.12,
        },
    )
    reward_jump_clearance = RewTerm(
        func=rewards.reward_jump_clearance,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_window_front": 0.65,
            "jump_window_back": -0.25,
            "safety_margin": 0.12,
        },
    )
    reward_crawl_clearance = RewTerm(
        func=rewards.reward_crawl_clearance,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
        },
    )
    reward_feet_clearance = RewTerm(
        func=rewards.reward_feet_clearance,
        weight=-.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "check_margin_x": 0.28,
            "check_margin_y": 0.85,
            "safety_margin": 0.05,
        },
    )
    reward_foot_symmetry = RewTerm(
        func=rewards.reward_foot_symmetry,
        weight=0.6,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_scale": 0.12},
    )
    reward_feet_air_time = RewTerm(
        func=rewards.reward_feet_air_time,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "min_air_time": 0.15,
        },
    )
    reward_lateral_deviation_penalty = RewTerm(
        func=rewards.reward_lateral_deviation_penalty,
        weight=-1.0,  # 负权重表示惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "lateral_threshold": 0.3,  # 横向偏移超过0.3米开始惩罚
            "penalty_scale": 2.0,  # 惩罚强度
        },
    )
    reward_successful_traversal = RewTerm(
        func=rewards.reward_successful_traversal,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "traversal_window": 0.55,
            "lateral_threshold": 0.35,
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    # 适度增加姿态/跟踪/接触约束，保持学生网络轻量化但不遗漏关键安全项
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    reward_ground_impact = RewTerm(
        func=rewards.reward_ground_impact,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_goal_progress = RewTerm(
        func=rewards.reward_goal_progress,
        weight=0.0,
        params={"parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques,
        weight=-3.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 诊断用：模式判定偏差与低杆爬行惩罚（小权重，仅供日志观察）
    reward_mode_mismatch = RewTerm(
        func=rewards.reward_mode_mismatch,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_low_crawl_penalty = RewTerm(
        func=rewards.reward_low_crawl_penalty,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TeacherRewardsCfg:
    """教师奖励：包含更丰富的约束/引导。

    - 在学生奖励基础上增加 DOF 误差、髋关节约束、姿态/角速度等项，强化机体姿态稳定性。
    - 更高的权重用于纠正不当接触/跌倒，提升教师策略的示范质量。
    """
    reward_alive = RewTerm(
        func=rewards.reward_alive,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_calf", ".*_thigh"])},
    )
    reward_height_guidance = RewTerm(
        func=rewards.reward_height_guidance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "target_height": 0.4,
            "speed_gate": 0.2,
        },
    )
    reward_jump_clearance = RewTerm(
        func=rewards.reward_jump_clearance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "jump_window_front": 0.65,
            "jump_window_back": -0.25,
            "safety_margin": 0.08,
            "speed_gate": 0.35,
        },
    )
    reward_crawl_clearance = RewTerm(
        func=rewards.reward_crawl_clearance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "speed_gate": 0.35,
        },
    )
    # reward_feet_clearance = RewTerm(
    #     func=rewards.reward_feet_clearance,
    #     weight=-0.,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "check_margin_x": 0.28,
    #         "check_margin_y": 0.85,
    #         "safety_margin": 0.05,
    #     },
    # )
    reward_foot_symmetry = RewTerm(
        func=rewards.reward_foot_symmetry,
        weight=.3,
        params={"asset_cfg": SceneEntityCfg("robot"), "height_scale": 0.12},
    )
    reward_feet_air_time = RewTerm(
        func=rewards.reward_feet_air_time,
        weight=1.,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "min_air_time": 0.15,
        },
    )
    reward_lateral_deviation_penalty = RewTerm(
        func=rewards.reward_lateral_deviation_penalty,
        weight=-1.5,  # 负权重表示惩罚，教师模型使用更强的惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "back_sense": 0.65,
            "detection_range": 1.4,
            "lateral_threshold": 0.3,  # 横向偏移超过0.3米开始惩罚
            "penalty_scale": 2.0,  # 惩罚强度
        },
    )
    reward_successful_traversal = RewTerm(
        func=rewards.reward_successful_traversal,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lane_half_width": 0.45,
            "traversal_window": 0.55,
            "lateral_threshold": 0.35,
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # reward_ground_impact = RewTerm(
    #     func=rewards.reward_ground_impact,
    #     weight=0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_mode_mismatch = RewTerm(
        func=rewards.reward_mode_mismatch,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    total_terminates = DoneTerm(
        func=terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """通用事件配置（复位、随机化等）。"""
    reset_root_state = EventTerm(
        func=events.reset_root_state,
        params={"offset": 3.0},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    # 物理随机化：摩擦、质量、质心偏移，提升策略鲁棒性
    physics_material = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)}
        },
    )
    random_camera_position = EventTerm(
        func=events.random_camera_position,
        mode="startup",
        params={"sensor_cfg": SceneEntityCfg("depth_camera"), "rot_noise_range": {"pitch": (-5, 5)}, "convention": "ros"},
    )
    # 场景扰动：周期性横向推力，防止策略在静态环境中过拟合
    push_by_setting_velocity = EventTerm(
        func=events.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        interval_range_s=(8.0, 8.0),
        is_global_time=True,
        mode="interval",
    )
    base_external_force_torque = EventTerm(
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class ActionsCfg:
    """动作配置（延迟的关节位置控制）。

    - action_delay_steps: 两帧管线延迟模拟通讯/执行滞后。
    - history_length: 叠加历史动作，帮助策略理解真实控制通道。
    """
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )
