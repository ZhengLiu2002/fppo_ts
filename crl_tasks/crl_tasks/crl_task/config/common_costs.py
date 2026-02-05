# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as CostTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from crl_isaaclab.envs.mdp import constraints as mdp_constraints
from crl_tasks.crl_task.config.galileo.config_summary import ConfigSummary


LEG_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    preserve_order=True,
)

FOOT_BODY_NAMES = [".*_foot"]
THIGH_BODY_NAMES = [".*_thigh"]


@configclass
class TeacherCostsCfg:
    """Cost terms mirrored from FPPO CMDP constraints (teacher)."""

    prob_joint_pos = CostTerm(
        func=mdp_constraints.joint_pos_prob_constraint,
        weight=ConfigSummary.cost.prob_joint_pos.weight,
        params={
            "margin": ConfigSummary.cost.prob_joint_pos.margin,
            "limit": ConfigSummary.cost.prob_joint_pos.limit,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )
    prob_joint_vel = CostTerm(
        func=mdp_constraints.joint_vel_prob_constraint,
        weight=ConfigSummary.cost.prob_joint_vel.weight,
        params={
            "limit": ConfigSummary.cost.prob_joint_vel.velocity_limit,
            "cost_limit": ConfigSummary.cost.prob_joint_vel.limit,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )
    prob_joint_torque = CostTerm(
        func=mdp_constraints.joint_torque_prob_constraint,
        weight=ConfigSummary.cost.prob_joint_torque.weight,
        params={
            "limit": ConfigSummary.cost.prob_joint_torque.torque_limit,
            "cost_limit": ConfigSummary.cost.prob_joint_torque.limit,
            "asset_cfg": LEG_JOINT_CFG,
        },
    )
    prob_body_contact = CostTerm(
        func=mdp_constraints.body_contact_prob_constraint,
        weight=ConfigSummary.cost.prob_body_contact.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": FOOT_BODY_NAMES,
            "threshold": ConfigSummary.cost.prob_body_contact.contact_force_threshold,
            "limit": ConfigSummary.cost.prob_body_contact.limit,
        },
    )
    prob_com_frame = CostTerm(
        func=mdp_constraints.com_frame_prob_constraint,
        weight=ConfigSummary.cost.prob_com_frame.weight,
        params={
            "height_range": ConfigSummary.cost.prob_com_frame.height_range,
            "max_angle_rad": ConfigSummary.cost.prob_com_frame.max_angle_rad,
            "cost_limit": ConfigSummary.cost.prob_com_frame.limit,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "height_offset": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    prob_gait_pattern = CostTerm(
        func=mdp_constraints.gait_pattern_prob_constraint,
        weight=ConfigSummary.cost.prob_gait_pattern.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": FOOT_BODY_NAMES,
            "gait_frequency": ConfigSummary.cost.prob_gait_pattern.gait_frequency,
            "phase_offsets": ConfigSummary.cost.prob_gait_pattern.phase_offsets,
            "stance_ratio": ConfigSummary.cost.prob_gait_pattern.stance_ratio,
            "contact_threshold": ConfigSummary.cost.prob_gait_pattern.contact_force_threshold,
            "command_name": "base_velocity",
            "min_frequency": ConfigSummary.cost.prob_gait_pattern.min_frequency,
            "max_frequency": ConfigSummary.cost.prob_gait_pattern.max_frequency,
            "max_command_speed": ConfigSummary.cost.prob_gait_pattern.max_command_speed,
            "frequency_scale": ConfigSummary.cost.prob_gait_pattern.frequency_scale,
            "min_command_speed": ConfigSummary.cost.prob_gait_pattern.min_command_speed,
            "min_base_speed": ConfigSummary.cost.prob_gait_pattern.min_base_speed,
            "asset_cfg": SceneEntityCfg("robot"),
            "limit": ConfigSummary.cost.prob_gait_pattern.limit,
        },
    )
    orthogonal_velocity = CostTerm(
        func=mdp_constraints.orthogonal_velocity_constraint,
        weight=ConfigSummary.cost.orthogonal_velocity.weight,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": ConfigSummary.cost.orthogonal_velocity.limit},
    )
    contact_velocity = CostTerm(
        func=mdp_constraints.contact_velocity_constraint,
        weight=ConfigSummary.cost.contact_velocity.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": FOOT_BODY_NAMES,
            "contact_threshold": ConfigSummary.cost.contact_velocity.contact_force_threshold,
            "limit": ConfigSummary.cost.contact_velocity.limit,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance = CostTerm(
        func=mdp_constraints.foot_clearance_constraint,
        weight=ConfigSummary.cost.foot_clearance.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": FOOT_BODY_NAMES,
            "min_height": ConfigSummary.cost.foot_clearance.min_height,
            "height_offset": 0.0,
            "contact_threshold": ConfigSummary.cost.contact_velocity.contact_force_threshold,
            "gait_frequency": ConfigSummary.cost.prob_gait_pattern.gait_frequency,
            "phase_offsets": ConfigSummary.cost.prob_gait_pattern.phase_offsets,
            "stance_ratio": ConfigSummary.cost.prob_gait_pattern.stance_ratio,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "limit": ConfigSummary.cost.foot_clearance.limit,
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "min_command_speed": ConfigSummary.cost.foot_clearance.min_command_speed,
            "min_base_speed": ConfigSummary.cost.foot_clearance.min_base_speed,
        },
    )
    foot_height_limit = CostTerm(
        func=mdp_constraints.foot_height_limit_constraint,
        weight=ConfigSummary.cost.foot_height_limit.weight,
        params={
            "foot_body_names": FOOT_BODY_NAMES,
            "height_offset": 0.0,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "limit": ConfigSummary.cost.foot_height_limit.limit,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    symmetric = CostTerm(
        func=mdp_constraints.symmetric_constraint,
        weight=ConfigSummary.cost.symmetric.weight,
        params={
            "joint_pair_indices": ConfigSummary.cost.symmetric.joint_pairs,
            "action_pair_indices": ConfigSummary.cost.symmetric.joint_pairs,
            "asset_cfg": SceneEntityCfg("robot"),
            "include_actions": True,
            "command_name": ConfigSummary.cost.symmetric.command_name,
            "min_command_speed": ConfigSummary.cost.symmetric.min_command_speed,
            "min_base_speed": ConfigSummary.cost.symmetric.min_base_speed,
            "limit": ConfigSummary.cost.symmetric.limit,
        },
    )
    base_contact_force = CostTerm(
        func=mdp_constraints.base_contact_force_constraint,
        weight=ConfigSummary.cost.base_contact_force.weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "body_names": THIGH_BODY_NAMES,
            "threshold": ConfigSummary.cost.base_contact_force.contact_force_threshold,
            "limit": ConfigSummary.cost.base_contact_force.limit,
        },
    )


@configclass
class StudentCostsCfg:
    """Simple proprioceptive constraints for the blind student."""

    prob_joint_pos = CostTerm(
        func=mdp_constraints.joint_pos_prob_constraint,
        weight=1.0,
        params={"margin": -0.05, "limit": 1.0, "asset_cfg": LEG_JOINT_CFG},
    )
    prob_joint_vel = CostTerm(
        func=mdp_constraints.joint_vel_prob_constraint,
        weight=1.0,
        params={"limit": 15.0, "cost_limit": 1.0, "asset_cfg": LEG_JOINT_CFG},
    )
    prob_joint_torque = CostTerm(
        func=mdp_constraints.joint_torque_prob_constraint,
        weight=1.0,
        params={"limit": 120.0, "cost_limit": 1.0, "asset_cfg": LEG_JOINT_CFG},
    )


# Backwards compatibility alias.
CostsCfg = TeacherCostsCfg
