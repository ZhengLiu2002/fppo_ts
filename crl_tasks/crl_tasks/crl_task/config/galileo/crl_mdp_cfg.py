from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from crl_isaaclab.envs.mdp.crl_actions import DelayedJointPositionActionCfg
import isaaclab.envs.mdp as mdp
from crl_isaaclab.envs.mdp import terminations, rewards, events, crl_commands, curriculums, observations as crl_obs
from crl_tasks.crl_task.config.common_costs import (
    TeacherCostsCfg,
    StudentCostsCfg,
    LEG_JOINT_CFG,
)
from .config_summary import ConfigSummary


@configclass
class CommandsCfg:
    """前进速度指令（针对直线跑道栏杆）

    - 仅在 X 方向采样 0.6~1.2 m/s，保持训练聚焦在跨栏核心动作。
    - heading 控制设置轻微偏移范围，利于生成左右对称的轨迹。
    """

    base_velocity = crl_commands.CRLCommandCfg(
        asset_name="robot",
        resampling_time_range=ConfigSummary.command.resampling_time_range,
        lin_x_level=ConfigSummary.command.lin_x_level,
        max_lin_x_level=ConfigSummary.command.max_lin_x_level,
        lin_x_level_step=ConfigSummary.command.lin_x_level,
        ang_z_level=ConfigSummary.command.ang_z_level,
        max_ang_z_level=ConfigSummary.command.max_ang_z_level,
        ang_z_level_step=ConfigSummary.command.ang_z_level,
        min_abs_lin_vel_x=ConfigSummary.command.min_abs_lin_vel_x,
        min_abs_lin_vel_y=ConfigSummary.command.min_abs_lin_vel_y,
        heading_control_stiffness=ConfigSummary.command.heading_control_stiffness,
        heading_command=True,
        rel_heading_envs=ConfigSummary.command.default.heading_command_prob,
        rel_standing_envs=ConfigSummary.command.default.standing_command_prob,
        align_to_crl_goal=False,
        ranges=crl_commands.CRLCommandCfg.Ranges(
            lin_vel_x=ConfigSummary.command.default.lin_vel_x,
            lin_vel_y=ConfigSummary.command.default.lin_vel_y,
            ang_vel_z=ConfigSummary.command.default.ang_vel_z,
            heading=ConfigSummary.command.default.heading,
            heading_command_prob=ConfigSummary.command.default.heading_command_prob,
            yaw_command_prob=ConfigSummary.command.default.yaw_command_prob,
            standing_command_prob=ConfigSummary.command.default.standing_command_prob,
            start_curriculum_lin_x=ConfigSummary.command.default.start_curriculum_lin_x,
            start_curriculum_ang_z=ConfigSummary.command.default.start_curriculum_ang_z,
            max_curriculum_lin_x=ConfigSummary.command.default.max_curriculum_lin_x,
            max_curriculum_ang_z=ConfigSummary.command.default.max_curriculum_ang_z,
        ),
        clips=crl_commands.CRLCommandCfg.Clips(
            lin_vel_clip=ConfigSummary.command.clips["lin_vel_clip"],
            ang_vel_clip=ConfigSummary.command.clips["ang_vel_clip"],
        ),
    )


@configclass
@configclass
class TeacherObservationsCfg:
    """FPPO 风格观测：base/imu/joint/command。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5}
        )
        base_com = ObsTerm(func=crl_obs.base_com, params={"body_name": "base_link"})
        base_mass = ObsTerm(func=crl_obs.base_mass, params={"body_name": "base_link"})
        ground_friction = ObsTerm(func=crl_obs.ground_friction)
    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5}
        )
        base_com = ObsTerm(func=crl_obs.base_com, params={"body_name": "base_link"})
        base_mass = ObsTerm(func=crl_obs.base_mass, params={"body_name": "base_link"})
        ground_friction = ObsTerm(func=crl_obs.ground_friction)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StudentObservationsCfg:
    """学生（蒸馏/学生网络）观测：FPPO 风格（盲走）。"""
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        base_com = ObsTerm(func=crl_obs.base_com, params={"body_name": "base_link"})
        base_mass = ObsTerm(func=crl_obs.base_mass, params={"body_name": "base_link"})
        history = ObsTerm(
            func=crl_obs.PolicyHistory,
            params={
                "history_length": 50,
                "include_base_lin_vel": False,
                "command_name": "base_velocity",
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class StudentRewardsCfg:
    """FPPO 奖励配置（盲走，不使用外感知传感器）。"""
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_exp,
        weight=ConfigSummary.reward.track_lin_vel_xy_exp.weight,
        params={
            "command_name": ConfigSummary.reward.track_lin_vel_xy_exp.command_name,
            "std": ConfigSummary.reward.track_lin_vel_xy_exp.std,
            "min_command_speed": ConfigSummary.reward.track_lin_vel_xy_exp.min_command_speed,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=rewards.track_ang_vel_z_exp,
        weight=ConfigSummary.reward.track_ang_vel_z_exp.weight,
        params={
            "command_name": ConfigSummary.reward.track_ang_vel_z_exp.command_name,
            "std": ConfigSummary.reward.track_ang_vel_z_exp.std,
            "min_command_speed": ConfigSummary.reward.track_ang_vel_z_exp.min_command_speed,
        },
    )
    flat_orientation_l2 = RewTerm(
        func=rewards.flat_orientation_l2, weight=ConfigSummary.reward.flat_orientation_l2.weight
    )
    joint_torques_l2 = RewTerm(
        func=rewards.joint_torques_l2, weight=ConfigSummary.reward.joint_torques_l2.weight
    )
    joint_vel_l2 = RewTerm(func=rewards.joint_vel_l2, weight=ConfigSummary.reward.joint_vel_l2.weight)
    joint_power_distribution = RewTerm(
        func=rewards.joint_power_distribution,
        weight=ConfigSummary.reward.joint_power_distribution.weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    joint_acc_l2 = RewTerm(func=rewards.joint_acc_l2, weight=ConfigSummary.reward.joint_acc_l2.weight)
    dof_error_l2 = RewTerm(
        func=rewards.dof_error_l2,
        weight=ConfigSummary.reward.dof_error_l2.weight,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    hip_pos_l2 = RewTerm(
        func=rewards.hip_pos_l2,
        weight=ConfigSummary.reward.hip_pos_l2.weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=ConfigSummary.reward.action_rate_l2.weight)
    action_smoothness_l2 = RewTerm(
        func=rewards.action_smoothness_l2, weight=ConfigSummary.reward.action_smoothness_l2.weight
    )
    lin_vel_z_l2 = RewTerm(func=rewards.lin_vel_z_l2, weight=ConfigSummary.reward.lin_vel_z_l2.weight)
    ang_vel_xy_l2 = RewTerm(func=rewards.ang_vel_xy_l2, weight=ConfigSummary.reward.ang_vel_xy_l2.weight)


@configclass
class TeacherRewardsCfg:
    """FPPO 奖励配置（与 FPPO 基线一致）。"""
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_exp,
        weight=ConfigSummary.reward.track_lin_vel_xy_exp.weight,
        params={
            "command_name": ConfigSummary.reward.track_lin_vel_xy_exp.command_name,
            "std": ConfigSummary.reward.track_lin_vel_xy_exp.std,
            "min_command_speed": ConfigSummary.reward.track_lin_vel_xy_exp.min_command_speed,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=rewards.track_ang_vel_z_exp,
        weight=ConfigSummary.reward.track_ang_vel_z_exp.weight,
        params={
            "command_name": ConfigSummary.reward.track_ang_vel_z_exp.command_name,
            "std": ConfigSummary.reward.track_ang_vel_z_exp.std,
            "min_command_speed": ConfigSummary.reward.track_ang_vel_z_exp.min_command_speed,
        },
    )
    flat_orientation_l2 = RewTerm(
        func=rewards.flat_orientation_l2, weight=ConfigSummary.reward.flat_orientation_l2.weight
    )
    base_height_l2_fix = RewTerm(
        func=rewards.base_height_l2_fix,
        weight=ConfigSummary.reward.base_height_l2_fix.weight,
        params={
            "target_height": ConfigSummary.reward.base_height_l2_fix.target_height,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    joint_torques_l2 = RewTerm(
        func=rewards.joint_torques_l2, weight=ConfigSummary.reward.joint_torques_l2.weight
    )
    joint_vel_l2 = RewTerm(func=rewards.joint_vel_l2, weight=ConfigSummary.reward.joint_vel_l2.weight)
    joint_power_distribution = RewTerm(
        func=rewards.joint_power_distribution,
        weight=ConfigSummary.reward.joint_power_distribution.weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    joint_acc_l2 = RewTerm(func=rewards.joint_acc_l2, weight=ConfigSummary.reward.joint_acc_l2.weight)
    dof_error_l2 = RewTerm(
        func=rewards.dof_error_l2,
        weight=ConfigSummary.reward.dof_error_l2.weight,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    hip_pos_l2 = RewTerm(
        func=rewards.hip_pos_l2,
        weight=ConfigSummary.reward.hip_pos_l2.weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=ConfigSummary.reward.action_rate_l2.weight)
    action_smoothness_l2 = RewTerm(
        func=rewards.action_smoothness_l2, weight=ConfigSummary.reward.action_smoothness_l2.weight
    )
    lin_vel_z_l2 = RewTerm(func=rewards.lin_vel_z_l2, weight=ConfigSummary.reward.lin_vel_z_l2.weight)
    ang_vel_xy_l2 = RewTerm(func=rewards.ang_vel_xy_l2, weight=ConfigSummary.reward.ang_vel_xy_l2.weight)
    feet_air_time = RewTerm(
        func=rewards.feet_air_time,
        weight=ConfigSummary.reward.feet_air_time.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": ConfigSummary.reward.feet_air_time.command_name,
            "threshold": ConfigSummary.reward.feet_air_time.threshold,
        },
    )
    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=ConfigSummary.reward.feet_slide.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    load_sharing = RewTerm(
        func=rewards.load_sharing,
        weight=ConfigSummary.reward.load_sharing.weight,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    undesired_contacts = RewTerm(
        func=rewards.undesired_contacts,
        weight=ConfigSummary.reward.undesired_contacts.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": ConfigSummary.reward.undesired_contacts.threshold,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=terminations.time_out,
        time_out=True,
    )
    bad_orientation = DoneTerm(
        func=terminations.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "roll_limit": 1.3,  # 从1.2放宽到1.4，减少过早终止（当前47%终止率太高）
            "pitch_limit": 1.3,  # 从1.2放宽到1.4，减少过早终止（当前47%终止率太高）
        },
    )
    body_contact = DoneTerm(
        func=terminations.body_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )


@configclass
class EventCfg:
    """域随机化配置（按照优化后的清单）。
    
    一、机身/动力学（reset 时）
    - 机身质量：randomize_base_mass
    - 机身质心：randomize_base_com
    - 摩擦系数：physics_material（静摩擦、动摩擦、恢复系数）
    
    二、重置位姿/速度（reset 时）
    - 基座位姿：reset_base_pose
    - 基座速度：reset_base_velocity（仅线速度随机）
    
    三、腿部关节（reset 时）
    - 腿部关节位置：reset_leg_joints
    
    四、执行器增益（reset 时）
    - Kp/Kd：randomize_actuator_gains
    - Kt：需要扩展功能（当前不支持）
    
    五、外力扰动（interval，训练中周期施加）
    - 速度扰动：push_robot_vel
    - 力矩扰动：push_robot_torque
    """
    
    def __post_init__(self):
        """在初始化后处理配置，移除不应该传递给 __call__ 的参数。"""
        # restitution_range 在 randomize_rigid_body_material 的 __init__ 中使用，
        # 但不应该作为 __call__ 的参数传递
        if hasattr(self, 'physics_material') and self.physics_material is not None:
            if hasattr(self.physics_material, 'params') and 'restitution_range' in self.physics_material.params:
                # 保存 restitution_range 的值，但不在 params 中传递
                # 它会在 __init__ 时从 params 读取
                pass  # 保持原样，因为事件管理器会在创建实例前检查
    
    # ========== 一、机身/动力学（reset 时） ==========
    # 机身质量随机化（对 base_link 质量做加性随机）
    randomize_base_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": ConfigSummary.event.randomize_base_mass.mass_distribution_params,
            "operation": ConfigSummary.event.randomize_base_mass.operation,
        },
    )
    
    # 机身质心随机化（对 base_link 的 CoM 在 body 系下随机偏移）
    randomize_base_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": ConfigSummary.event.randomize_base_com.com_range,
        },
    )
    
    # 物理材质随机化（静摩擦、动摩擦、恢复系数）
    # 注意：restitution_range 在 __init__ 中使用，但不作为 __call__ 的参数
    # 我们需要在配置后手动处理，但由于事件管理器在创建实例前检查参数，
    # 我们需要使用一个包装函数或者修改 __call__ 签名来接受但不使用这个参数
    # 暂时移除 make_consistent，因为它会导致参数检查失败
    physics_material = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": ConfigSummary.event.physics_material.friction_range,
            "restitution_range": ConfigSummary.event.physics_material.restitution_range,
            "num_buckets": ConfigSummary.event.physics_material.num_buckets,
        },
    )
    
    # ========== 二、重置位姿/速度（reset 时） ==========
    # 基座位姿随机化
    reset_base_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": ConfigSummary.event.reset_base_pose.pose_range,
            "velocity_range": ConfigSummary.event.reset_base_pose.velocity_range,
        },
        mode="reset",
    )
    
    # ========== 三、腿部关节（reset 时） ==========
    # 腿部关节位置随机化（仅对 12 个腿关节）
    reset_leg_joints = EventTerm(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": LEG_JOINT_CFG,
            "position_range": ConfigSummary.event.reset_leg_joints.position_range,
            "velocity_range": ConfigSummary.event.reset_leg_joints.velocity_range,
        },
    )
    
    # ========== 四、执行器增益（reset 时） ==========
    # Kp/Kd 增益随机化（对腿 + 臂执行器）
    randomize_actuator_kp_kd_gains = EventTerm(
        func=events.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),  # 所有关节
            "stiffness_distribution_params": ConfigSummary.event.randomize_actuator_kp_kd_gains.stiffness_distribution_params,
            "damping_distribution_params": ConfigSummary.event.randomize_actuator_kp_kd_gains.damping_distribution_params,
            "operation": ConfigSummary.event.randomize_actuator_kp_kd_gains.operation,
            "distribution": ConfigSummary.event.randomize_actuator_kp_kd_gains.distribution,
        },
    )
    
    # Kt 增益随机化（需要扩展功能，当前不支持）
    # 注意：当前代码库中 randomize_actuator_gains 不支持 Kt 随机化
    # 如果需要此功能，需要扩展 events.randomize_actuator_gains 函数
    # randomize_actuator_kt_gains = EventTerm(
    #     func=events.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "kt_distribution_params": (0.8, 1.2),  # Kt 乘性
    #         "operation": "scale",
    #     },
    # )
    
    # ========== 五、外力扰动（interval，训练中周期施加） ==========
    # 速度扰动（对基座施加瞬时速度扰动）
    push_robot_vel = EventTerm(
        func=events.push_by_setting_velocity,
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # push_by_setting_velocity 作用于整个 asset
            "velocity_range": ConfigSummary.event.push_robot_vel.velocity_range,
        },
        interval_range_s=ConfigSummary.event.push_robot_vel.interval_range_s,
        is_global_time=ConfigSummary.event.push_robot_vel.is_global_time,
        mode="interval",
    )
    
    # 力矩扰动（对 base_link 施加外力矩）
    push_robot_torque = EventTerm(
        func=apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": ConfigSummary.event.push_robot_torque.force_range,
            "torque_range": ConfigSummary.event.push_robot_torque.torque_range,
        },
        interval_range_s=ConfigSummary.event.push_robot_torque.interval_range_s,
        is_global_time=ConfigSummary.event.push_robot_torque.is_global_time,
        mode="interval",
    )
    
    # 保留相机随机化（如果需要）
    random_camera_position = EventTerm(
        func=events.random_camera_position,
        mode="startup",
        params={"sensor_cfg": SceneEntityCfg("depth_camera"), "rot_noise_range": {"pitch": (-5, 5)}, "convention": "ros"},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=curriculums.terrain_levels_vel,
        params={"episodes_per_level": ConfigSummary.curriculum.episodes_per_level},
    )
    lin_vel_x_command_threshold = CurrTerm(
        func=curriculums.lin_vel_x_command_threshold,
        params={"episodes_per_level": ConfigSummary.curriculum.episodes_per_level},
    )
    ang_vel_z_command_threshold = CurrTerm(
        func=curriculums.ang_vel_z_command_threshold,
        params={"episodes_per_level": ConfigSummary.curriculum.episodes_per_level},
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
        scale=0.5,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=1,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )
