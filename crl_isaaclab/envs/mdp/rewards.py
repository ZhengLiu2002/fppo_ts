# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    low_speed_threshold: float = 0.4,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # reduce stepping reward at low command speeds (avoid forcing steps during slow/stand)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    speed_scale = torch.clamp(cmd_norm / max(low_speed_threshold, 1.0e-6), min=0.0, max=1.0)
    reward *= speed_scale
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Check dimensions. contacts is (num_envs, num_feet)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    # Check dimensions. body_vel is (num_envs, num_bodies, 2). 
    # We need to slice asset.data.body_lin_vel_w correctly corresponding to the feet body ids.
    # Note: asset_cfg used here may default to all bodies if not specified, 
    # but for this calculation we specifically need the feet velocities.
    # So we should probably use the same body_ids as the sensor_cfg if not explicitly provided in asset_cfg.
    
    # However, asset_cfg in the config is just SceneEntityCfg("robot"), which implies all bodies or default.
    # The error message "The size of tensor a (19) must match the size of tensor b (4)" suggests:
    # 'a' is body_vel (19 bodies in robot?)
    # 'b' is contacts (4 feet)
    
    # We must ensure we select the feet bodies from the robot asset.
    # We can reuse sensor_cfg.body_ids because that selector was used to get the contact forces for the feet.
    
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def load_sharing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 2,
    force_threshold: float = 1.0,
    var_scale: float = 1.0,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Reward even load sharing across feet using contact force distribution."""
    contact_sensor: ContactSensor | None = env.scene.sensors.get(sensor_cfg.name, None)
    if contact_sensor is None:
        return torch.zeros(env.scene.num_envs, device=env.device)

    if isinstance(sensor_cfg.body_ids, slice):
        return torch.zeros(env.scene.num_envs, device=env.device)

    num_feet = len(sensor_cfg.body_ids)
    if num_feet == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    min_contacts = max(int(min_contacts), 1)
    var_scale = max(float(var_scale), 0.0)

    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1).max(dim=1)[0]
    contact_mask = force_mag > force_threshold
    num_contacts = contact_mask.sum(dim=1)

    force_mag = torch.where(contact_mask, force_mag, torch.zeros_like(force_mag))
    total_force = force_mag.sum(dim=1, keepdim=True).clamp_min(eps)
    share = force_mag / total_force

    inv_contacts = 1.0 / num_contacts.clamp_min(1).float()
    target_share = inv_contacts.unsqueeze(1).expand_as(share)
    diff = (share - target_share) * contact_mask.float()
    var = (diff * diff).sum(dim=1) * inv_contacts

    reward = torch.clamp(1.0 - var_scale * var, min=0.0)
    reward = torch.where(num_contacts >= min_contacts, reward, torch.zeros_like(reward))
    return reward


def foothold_terrain_flatness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    variance_scale: float | None = None,
    sample_stride: int = 1,
    variance_kernel_size: int = 3,
    decay_k: float = 1.0,
    decay_sigma: float = 1.0,
    min_sigma: float = 1.0e-3,
    max_decay_exp: float = 5.0,
    use_derived_action: bool = True,
) -> torch.Tensor:
    """Penalize footholds on uneven terrain using local height variance."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None or len(asset_cfg.body_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]

    sensor = env.scene.sensors.get(sensor_cfg.name, None) if sensor_cfg is not None else None
    if not isinstance(sensor, RayCaster):
        return torch.zeros(env.scene.num_envs, device=env.device)

    pattern_cfg = getattr(sensor.cfg, "pattern_cfg", None)
    resolution = getattr(pattern_cfg, "resolution", None)
    size = getattr(pattern_cfg, "size", None)
    if resolution is None or size is None or resolution <= 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    size_x = float(size[0])
    size_y = float(size[1])
    num_x = int(round(size_x / resolution)) + 1
    num_y = int(round(size_y / resolution)) + 1
    stride = max(int(sample_stride), 1)

    ray_z = sensor.data.ray_hits_w[..., 2]
    ray_z = torch.where(torch.isfinite(ray_z), ray_z, torch.zeros_like(ray_z))
    if ray_z.shape[1] != num_x * num_y:
        return torch.zeros(env.scene.num_envs, device=env.device)

    derived_action = getattr(env, "derived_action", None) if use_derived_action else None
    if derived_action is not None:
        if derived_action.ndim == 2:
            if derived_action.shape[1] % 3 != 0:
                return torch.zeros(env.scene.num_envs, device=env.device)
            derived_action = derived_action.view(derived_action.shape[0], -1, 3)
        if derived_action.ndim != 3 or derived_action.shape[2] < 3:
            return torch.zeros(env.scene.num_envs, device=env.device)
        derived_action = derived_action.to(foot_pos_w.device)

        ordering = getattr(pattern_cfg, "ordering", "xy")
        if ordering == "yx":
            ray_z_2d = ray_z.view(ray_z.shape[0], num_x, num_y).transpose(1, 2)
        else:
            ray_z_2d = ray_z.view(ray_z.shape[0], num_y, num_x)

        k = max(int(variance_kernel_size), 1)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        kernel = torch.ones((1, 1, k, k), device=ray_z_2d.device, dtype=ray_z_2d.dtype) / float(k * k)
        ray_z_2d = ray_z_2d.unsqueeze(1)
        ray_z_pad = F.pad(ray_z_2d, (pad, pad, pad, pad), mode="replicate")
        mean = F.conv2d(ray_z_pad, kernel)
        mean_sq = F.conv2d(ray_z_pad.pow(2), kernel)
        var_map = (mean_sq - mean.pow(2)).squeeze(1)

        if stride > 1:
            var_map = var_map[:, ::stride, ::stride]

        x_coords = torch.linspace(
            -0.5 * size_x, 0.5 * size_x, num_x, device=var_map.device, dtype=var_map.dtype
        )
        y_coords = torch.linspace(
            -0.5 * size_y, 0.5 * size_y, num_y, device=var_map.device, dtype=var_map.dtype
        )
        if stride > 1:
            x_coords = x_coords[::stride]
            y_coords = y_coords[::stride]

        num_x_eff = x_coords.numel()
        num_y_eff = y_coords.numel()
        grid_x = x_coords.repeat(num_y_eff)
        grid_y = y_coords.repeat_interleave(num_x_eff)

        var_flat = var_map.reshape(var_map.shape[0], -1)
        if variance_scale is None:
            variance_scale = 1.0 / max(float(resolution), 1.0e-6) ** 2
        var_flat = var_flat * float(variance_scale)

        mu_b = derived_action[..., :2]
        sigma = torch.clamp(derived_action[..., 2], min=min_sigma)
        mu_b3 = torch.cat((mu_b, torch.zeros_like(mu_b[..., :1])), dim=-1)
        base_quat = sensor.data.quat_w.unsqueeze(1).expand(-1, mu_b3.shape[1], -1)
        base_pos = sensor.data.pos_w.unsqueeze(1)
        mu_w = quat_apply(base_quat, mu_b3) + base_pos
        rel = mu_w - base_pos
        alignment = getattr(sensor.cfg, "ray_alignment", "yaw")
        if alignment == "world":
            rel_local = rel
        else:
            rel_local = quat_apply_inverse(sensor.data.quat_w, rel)
        mu_xy = rel_local[..., :2]

        delta_x = grid_x.unsqueeze(0) - mu_xy[:, :, 0].unsqueeze(-1)
        delta_y = grid_y.unsqueeze(0) - mu_xy[:, :, 1].unsqueeze(-1)
        dist_sq = delta_x.pow(2) + delta_y.pow(2)
        sigma_sq = torch.clamp(sigma, min=min_sigma).pow(2).unsqueeze(-1)
        decay = torch.exp(-0.5 * dist_sq / sigma_sq)
        decay = decay / decay.sum(dim=-1, keepdim=True).clamp_min(min_sigma)
        decay = torch.clamp(decay, min=0.0, max=math.exp(max_decay_exp))

        weighted_var = (var_flat.unsqueeze(1) * decay).sum(dim=-1)
        penalty = weighted_var.mean(dim=1)
        return penalty

    # fallback: use local variance around actual foot positions
    ordering = getattr(pattern_cfg, "ordering", "xy")
    if ordering == "yx":
        ray_z_2d = ray_z.view(ray_z.shape[0], num_x, num_y).transpose(1, 2)
    else:
        ray_z_2d = ray_z.view(ray_z.shape[0], num_y, num_x)
    k = max(int(variance_kernel_size), 1)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = torch.ones((1, 1, k, k), device=ray_z_2d.device, dtype=ray_z_2d.dtype) / float(k * k)
    ray_z_2d = ray_z_2d.unsqueeze(1)
    ray_z_pad = F.pad(ray_z_2d, (pad, pad, pad, pad), mode="replicate")
    mean = F.conv2d(ray_z_pad, kernel)
    mean_sq = F.conv2d(ray_z_pad.pow(2), kernel)
    var_map = (mean_sq - mean.pow(2)).squeeze(1)

    if stride > 1:
        var_map = var_map[:, ::stride, ::stride]

    x_coords = torch.linspace(-0.5 * size_x, 0.5 * size_x, num_x, device=var_map.device, dtype=var_map.dtype)
    y_coords = torch.linspace(-0.5 * size_y, 0.5 * size_y, num_y, device=var_map.device, dtype=var_map.dtype)
    if stride > 1:
        x_coords = x_coords[::stride]
        y_coords = y_coords[::stride]

    num_x_eff = x_coords.numel()
    num_y_eff = y_coords.numel()
    x_grid = x_coords.view(1, 1, num_x_eff).expand(env.num_envs, num_y_eff, -1)
    y_grid = y_coords.view(1, num_y_eff, 1).expand(env.num_envs, -1, num_x_eff)

    foot_xy = foot_pos_w[..., :2]
    x_idx = torch.argmin(torch.abs(x_grid.unsqueeze(1) - foot_xy[:, :, 0].unsqueeze(-1)), dim=-1)
    y_idx = torch.argmin(torch.abs(y_grid.unsqueeze(1) - foot_xy[:, :, 1].unsqueeze(-1)), dim=-1)
    idx = y_idx * num_x_eff + x_idx
    var_flat = var_map.reshape(var_map.shape[0], -1)
    var_flat = var_flat.gather(1, idx)
    if variance_scale is None:
        variance_scale = 1.0 / max(float(resolution), 1.0e-6) ** 2
    penalty = var_flat.mean(dim=1) * float(variance_scale)
    return penalty


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def command_tracking_quadratic(
    env: ManagerBasedRLEnv,
    command_name: str,
    kappa_lin: float,
    kappa_ang: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Command tracking reward using a negative quadratic penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd = env.command_manager.get_command(command_name)
    lin_error = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    ang_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_w[:, 2])
    kappa_ang = kappa_lin if kappa_ang is None else kappa_ang
    return -(kappa_lin * lin_error + kappa_ang * ang_error)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power distribution."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.var(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_torque_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_mass: float | None = None,
    ref_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint torque using L2 norm squared."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(torque), dim=1)

    if ref_mass is None:
        return penalty

    scale_attr = "_torque_mass_scale"
    if not hasattr(env, scale_attr):
        masses = getattr(asset.data, "default_mass", None)
        if masses is None:
            setattr(env, scale_attr, ref_weight)
        else:
            robot_mass = masses[0].sum().item()
            scale = ref_weight * (ref_mass / max(robot_mass, 1.0e-6))
            setattr(env, scale_attr, scale)
    scale = getattr(env, scale_attr)
    return penalty * scale


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torque using L2 norm squared (alias for joint_torque_l2)."""
    return joint_torque_l2(env, asset_cfg=asset_cfg)


def amp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward AMP."""
    # extract the used quantities (to enable type-hinting)
    reward = torch.clamp(1 - (1 / 4) * torch.square(env.amp_out - 1), min=0)
    return reward.squeeze()


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward action smoothness."""
    actions = getattr(env, "actions_history", None)
    if actions is not None:
        diff = torch.square(actions.get_data_vec([0]) - 2 * actions.get_data_vec([1]) + actions.get_data_vec([2]))
        return torch.sum(diff, dim=1)
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev = getattr(env, "_prev_prev_action", prev_action)
    diff2 = action - 2 * prev_action + prev_prev
    return torch.sum(torch.square(diff2), dim=1)


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    action_diff_weight: float = 1.0,
    action_diff2_weight: float = 1.0,
    joint_vel_weight: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize action smoothness using first- and second-order differences."""
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev = getattr(env, "_prev_prev_action", prev_action)

    diff1 = action - prev_action
    diff2 = action - 2 * prev_action + prev_prev
    penalty = action_diff_weight * torch.sum(torch.square(diff1), dim=1)
    penalty += action_diff2_weight * torch.linalg.norm(diff2, dim=1)

    if joint_vel_weight > 0.0:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        penalty += joint_vel_weight * torch.sum(torch.square(joint_vel), dim=1)

    return penalty


def stand_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    return reward


def base_height_l2_fix(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        # 检查sensor数据是否包含inf或nan
        ray_hits = sensor.data.ray_hits_w[..., 2]
        ray_hits = torch.where(torch.isinf(ray_hits), 0.0, ray_hits)
        ray_hits = torch.where(torch.isnan(ray_hits), 0.0, ray_hits)
        adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_command_speed: float | None = None,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    if min_command_speed is not None:
        cmd_speed = torch.norm(command[:, :2], dim=1)
        reward = reward * (cmd_speed > min_command_speed).float()
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_command_speed: float | None = None,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if min_command_speed is not None:
        reward = reward * (torch.abs(command[:, 2]) > min_command_speed).float()
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions from default ones using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    return is_contact.float()


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they are close to limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute limits
    out_of_limits = -(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they are close to limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute limits
    out_of_limits = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(out_of_limits.clip(min=0.0), dim=1)

def hip_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize hip joint positions that deviate from default pose using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return torch.zeros(env.scene.num_envs, device=env.device)
    if not isinstance(joint_ids, slice) and len(joint_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)
    diff = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(diff), dim=1)


def dof_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all joint positions that deviate from default pose using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return torch.zeros(env.scene.num_envs, device=env.device)
    if not isinstance(joint_ids, slice) and len(joint_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)
    diff = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(diff), dim=1)
