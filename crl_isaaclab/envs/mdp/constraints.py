# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common constraint terms for CMDP-style training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _warn_once(env: ManagerBasedRLEnv, flag_name: str, message: str) -> None:
    if getattr(env, flag_name, False):
        return
    setattr(env, flag_name, True)
    print(f"[WARN] {message}")


def _get_joint_slice(asset_cfg: SceneEntityCfg | None) -> slice | list[int]:
    if asset_cfg is None or asset_cfg.joint_ids is None:
        return slice(None)
    return asset_cfg.joint_ids


def _zeros_like_env(env: ManagerBasedRLEnv, dtype: torch.dtype | None = None) -> torch.Tensor:
    device = getattr(env, "device", torch.device("cpu"))
    return torch.zeros(env.num_envs, device=device, dtype=dtype or torch.float32)


def _normalize_cost(cost: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is None:
        return cost
    limit_t = torch.as_tensor(limit, device=cost.device, dtype=cost.dtype)
    limit_t = torch.abs(limit_t)
    eps = torch.finfo(cost.dtype).eps if torch.is_floating_point(cost) else 1e-6
    limit_t = torch.clamp(limit_t, min=eps)
    return cost / limit_t


def _dynamic_limit_scale(
    env: ManagerBasedRLEnv,
    epsilon: float | None,
    k: float | None,
) -> float:
    """Compute curriculum scale phi(step) = (epsilon - 1) * exp(-k * step) + 1."""
    if epsilon is None:
        return 1.0
    eps = max(float(epsilon), 1.0e-6)
    if k is None:
        return eps
    k_val = max(float(k), 0.0)
    if k_val == 0.0:
        return eps
    step = float(getattr(env, "common_step_counter", getattr(env, "_sim_step_counter", 0)))
    phi = (eps - 1.0) * math.exp(-k_val * step) + 1.0
    return max(phi, 1.0e-6)


def _smoothstep_progress(
    env: ManagerBasedRLEnv,
    schedule_start_step: int | None,
    schedule_end_step: int | None,
) -> float:
    """Return a smoothed [0, 1] curriculum progress from global step."""
    if schedule_start_step is None or schedule_end_step is None:
        return 1.0
    start = float(schedule_start_step)
    end = float(schedule_end_step)
    step = float(getattr(env, "common_step_counter", getattr(env, "_sim_step_counter", 0)))
    if end <= start:
        return 1.0 if step >= end else 0.0
    u = (step - start) / (end - start)
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def _command_gate(
    env: ManagerBasedRLEnv,
    command_name: str | None,
    min_command_speed: float | None,
    min_base_speed: float | None,
    asset_cfg: SceneEntityCfg,
    device: torch.device,
    dtype: torch.dtype,
    max_abs_yaw_cmd: float | None = None,
) -> torch.Tensor | None:
    gate = None
    if command_name is not None and hasattr(env, "command_manager"):
        commands = env.command_manager.get_command(command_name)
        if commands is not None:
            cmd_speed = torch.linalg.norm(commands[:, :2], dim=1)
            if min_command_speed is not None:
                gate = cmd_speed >= min_command_speed
            else:
                gate = torch.ones_like(cmd_speed, dtype=torch.bool)
            if max_abs_yaw_cmd is not None:
                yaw_cmd = (
                    torch.abs(commands[:, 2])
                    if commands.shape[1] > 2
                    else torch.zeros_like(cmd_speed)
                )
                yaw_gate = yaw_cmd <= max_abs_yaw_cmd
                gate = yaw_gate if gate is None else (gate & yaw_gate)
    if min_base_speed is not None:
        asset: Articulation = env.scene[asset_cfg.name]
        base_speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
        base_gate = base_speed >= min_base_speed
        gate = base_gate if gate is None else (gate & base_gate)
    if gate is None:
        return None
    return gate.to(device=device, dtype=dtype)


def _resolve_gait_frequency(
    env: ManagerBasedRLEnv,
    command_name: str | None,
    base_frequency: float,
    min_frequency: float | None,
    max_frequency: float | None,
    max_command_speed: float | None,
    frequency_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    base_t = torch.as_tensor(base_frequency, device=device, dtype=dtype)
    if command_name is None or not hasattr(env, "command_manager"):
        return base_t.expand(env.num_envs)
    commands = env.command_manager.get_command(command_name)
    if commands is None:
        return base_t.expand(env.num_envs)
    speed = torch.linalg.norm(commands[:, :2], dim=1)

    if max_command_speed is not None and max_command_speed > 0.0:
        min_f = (
            base_t
            if min_frequency is None
            else torch.as_tensor(min_frequency, device=device, dtype=dtype)
        )
        max_f = (
            base_t
            if max_frequency is None
            else torch.as_tensor(max_frequency, device=device, dtype=dtype)
        )
        max_speed_t = torch.as_tensor(max_command_speed, device=device, dtype=dtype)
        ratio = torch.clamp(speed / max_speed_t, 0.0, 1.0)
        freq = min_f + (max_f - min_f) * ratio
    else:
        scale_t = torch.as_tensor(frequency_scale, device=device, dtype=dtype)
        freq = base_t + speed * scale_t

    if min_frequency is not None or max_frequency is not None:
        min_f = (
            base_t
            if min_frequency is None
            else torch.as_tensor(min_frequency, device=device, dtype=dtype)
        )
        max_f = (
            base_t
            if max_frequency is None
            else torch.as_tensor(max_frequency, device=device, dtype=dtype)
        )
        freq = torch.clamp(freq, min=min_f, max=max_f)

    return freq


def _resolve_command_speed(
    env: ManagerBasedRLEnv,
    command_name: str | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if command_name is None or not hasattr(env, "command_manager"):
        return None
    commands = env.command_manager.get_command(command_name)
    if commands is None:
        return None
    lin = torch.linalg.norm(commands[:, :2], dim=1)
    yaw = torch.abs(commands[:, 2]) if commands.shape[1] > 2 else torch.zeros_like(lin)
    return lin + yaw


def _resolve_base_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    if not hasattr(env, "scene"):
        return None
    if not hasattr(asset_cfg, "name"):
        return None
    scene_keys = env.scene.keys() if hasattr(env.scene, "keys") else []
    if asset_cfg.name not in scene_keys:
        return None
    try:
        asset: Articulation = env.scene[asset_cfg.name]
    except KeyError:
        return None
    lin_vel = getattr(asset.data, "root_lin_vel_w", None)
    ang_vel = getattr(asset.data, "root_ang_vel_w", None)
    if lin_vel is None or ang_vel is None:
        return None
    lin = torch.linalg.norm(lin_vel[:, :2], dim=1)
    yaw = torch.abs(ang_vel[:, 2])
    return lin + yaw


def _resolve_episode_phase_time(
    env: ManagerBasedRLEnv,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Per-environment elapsed time within the current episode."""
    if dt <= 0.0:
        return torch.zeros(env.num_envs, device=device, dtype=dtype)

    episode_len = getattr(env, "episode_length_buf", None)
    if episode_len is not None and torch.is_tensor(episode_len):
        # episode_length_buf is incremented before cost computation; subtract one step
        # so each environment starts its phase from t=0 after reset.
        steps = episode_len.to(device=device, dtype=dtype) - 1.0
        steps = torch.clamp(steps, min=0.0)
        return steps * dt

    global_step = float(getattr(env, "_sim_step_counter", 0))
    return torch.full((env.num_envs,), global_step * dt, device=device, dtype=dtype)


def _terrain_height_at_points(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    points_w: torch.Tensor,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    ray_hits = getattr(sensor.data, "ray_hits_w", None)
    if ray_hits is None:
        _warn_once(
            env, "_warn_missing_ray_hits", "Ray hits not available; using zero terrain height."
        )
        return torch.zeros(points_w.shape[:2], device=points_w.device, dtype=points_w.dtype)

    invalid = torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1)
    ray_xy = ray_hits[..., :2]
    if invalid.any():
        far = torch.full_like(ray_xy, 1.0e6)
        ray_xy = torch.where(invalid.unsqueeze(-1), far, ray_xy)
    foot_xy = points_w[..., :2]
    diff = foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)
    dist2 = torch.sum(diff * diff, dim=-1)
    idx = torch.argmin(dist2, dim=2)
    ray_z = ray_hits[..., 2]
    if invalid.any():
        ray_z = torch.where(invalid, torch.zeros_like(ray_z), ray_z)
    ray_z_exp = ray_z.unsqueeze(1).expand(-1, foot_xy.shape[1], -1)
    return torch.gather(ray_z_exp, 2, idx.unsqueeze(-1)).squeeze(-1)


def _foot_heights_relative(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    foot_ids: list[int],
    terrain_sensor_cfg: SceneEntityCfg | None,
    height_offset: float,
) -> torch.Tensor:
    foot_pos_w = asset.data.body_pos_w[:, foot_ids]
    foot_heights = foot_pos_w[:, :, 2]
    if terrain_sensor_cfg is not None:
        terrain_heights = _terrain_height_at_points(env, terrain_sensor_cfg, foot_pos_w)
        return foot_heights - terrain_heights
    if height_offset != 0.0:
        return foot_heights - height_offset
    return foot_heights


def constraint_joint_pos(
    env: ManagerBasedRLEnv,
    margin: float = 0.3,
    joint_pos_window: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of joints violating soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_pos = asset.data.joint_pos
    if joint_pos is None:
        _warn_once(
            env,
            "_warn_missing_joint_pos",
            "Joint positions not available; joint_pos constraint disabled.",
        )
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_pos = joint_pos[:, joint_ids]

    limits = getattr(asset.data, "soft_joint_pos_limits", None)
    if limits is None:
        _warn_once(
            env,
            "_warn_missing_soft_joint_limits",
            "soft_joint_pos_limits not found; joint_pos constraint disabled.",
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    limits = limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
    if limits.ndim == 2:
        if not isinstance(joint_ids, slice):
            limits = limits[joint_ids]
        limits = limits.unsqueeze(0).expand(joint_pos.shape[0], -1, -1)
    elif limits.ndim == 3:
        if not isinstance(joint_ids, slice):
            limits = limits[:, joint_ids]
        if limits.shape[0] == 1 and joint_pos.shape[0] > 1:
            limits = limits.expand(joint_pos.shape[0], -1, -1)
        elif limits.shape[0] != joint_pos.shape[0]:
            _warn_once(
                env,
                "_warn_soft_joint_limits_shape",
                "soft_joint_pos_limits shape mismatch; joint_pos constraint disabled.",
            )
            return _zeros_like_env(env, dtype=joint_pos.dtype)
    else:
        _warn_once(
            env,
            "_warn_soft_joint_limits_ndim",
            "soft_joint_pos_limits has unexpected shape; joint_pos constraint disabled.",
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    margin_t = torch.as_tensor(margin, device=joint_pos.device, dtype=joint_pos.dtype)
    soft_lower = limits[..., 0] - margin_t
    soft_upper = limits[..., 1] + margin_t
    lower = soft_lower.clone()
    upper = soft_upper.clone()

    if joint_pos_window:
        all_joint_names = getattr(asset.data, "joint_names", None)
        if all_joint_names is None:
            _warn_once(
                env,
                "_warn_missing_joint_names_for_window",
                "joint_pos_window provided but joint names are unavailable; using soft limits only.",
            )
        else:
            if isinstance(joint_ids, slice):
                selected_joint_names = list(all_joint_names)
            else:
                selected_joint_names = [all_joint_names[idx] for idx in joint_ids]

            has_reversed_bounds = False
            for local_idx, joint_name in enumerate(selected_joint_names):
                bounds = joint_pos_window.get(joint_name)
                if bounds is None:
                    continue
                raw_low, raw_high = float(bounds[0]), float(bounds[1])
                bound_low = min(raw_low, raw_high)
                bound_high = max(raw_low, raw_high)
                if raw_low > raw_high:
                    has_reversed_bounds = True

                bound_low_t = torch.as_tensor(
                    bound_low, device=joint_pos.device, dtype=joint_pos.dtype
                )
                bound_high_t = torch.as_tensor(
                    bound_high, device=joint_pos.device, dtype=joint_pos.dtype
                )
                lower[:, local_idx] = torch.maximum(lower[:, local_idx], bound_low_t)
                upper[:, local_idx] = torch.minimum(upper[:, local_idx], bound_high_t)

            if has_reversed_bounds:
                _warn_once(
                    env,
                    "_warn_joint_pos_window_order",
                    "Detected reversed [upper, lower] entries in joint_pos_window; auto-corrected.",
                )

            invalid_window = lower > upper
            if invalid_window.any():
                _warn_once(
                    env,
                    "_warn_joint_pos_window_conflict",
                    "joint_pos_window conflicts with soft limits for some joints; falling back to soft limits on those joints.",
                )
                lower = torch.where(invalid_window, soft_lower, lower)
                upper = torch.where(invalid_window, soft_upper, upper)

    violation = (joint_pos < lower) | (joint_pos > upper)
    return violation.float().mean(dim=1)


def joint_pos_prob_constraint(
    env: ManagerBasedRLEnv,
    margin: float = 0.0,
    joint_pos_window: dict[str, tuple[float, float]] | None = None,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cost = constraint_joint_pos(
        env,
        margin=margin,
        joint_pos_window=joint_pos_window,
        asset_cfg=asset_cfg,
    )
    return _normalize_cost(cost, limit)


def joint_vel_prob_constraint(
    env: ManagerBasedRLEnv,
    limit: float = 50.0,
    soft_ratio: float = 1.0,
    cost_limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cost = constraint_joint_vel(
        env,
        limit=limit,
        soft_ratio=soft_ratio,
        asset_cfg=asset_cfg,
    )
    return _normalize_cost(cost, cost_limit)


def joint_torque_prob_constraint(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    soft_ratio: float = 1.0,
    cost_limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cost = constraint_joint_torque(
        env,
        limit=limit,
        soft_ratio=soft_ratio,
        asset_cfg=asset_cfg,
    )
    return _normalize_cost(cost, cost_limit)


def body_contact_prob_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    threshold: float = 1.0,
    limit: float | None = None,
) -> torch.Tensor:
    """Fraction of non-foot bodies exceeding contact force threshold."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    all_ids, all_names = contact_sensor.find_bodies([".*"], preserve_order=True)
    if not all_ids:
        return _zeros_like_env(env)
    foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    non_foot_ids = [i for i in all_ids if i not in foot_ids]
    if not non_foot_ids:
        return _zeros_like_env(env)
    net_forces = contact_sensor.data.net_forces_w_history
    force_mag = torch.norm(net_forces, dim=-1).max(dim=1)[0][:, non_foot_ids]
    threshold_t = torch.as_tensor(threshold, device=force_mag.device, dtype=force_mag.dtype)
    violation = force_mag > threshold_t
    cost = violation.float().mean(dim=1)
    return _normalize_cost(cost, limit)


def com_frame_prob_constraint(
    env: ManagerBasedRLEnv,
    height_range: tuple[float, float] = (0.2, 0.8),
    max_angle_rad: float = 0.35,
    cost_limit: float | None = None,
    limit_relax_epsilon: float | None = None,
    limit_relax_k: float | None = None,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    height_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_range_start: tuple[float, float] | None = None,
    height_range_end: tuple[float, float] | None = None,
    max_angle_rad_start: float | None = None,
    max_angle_rad_end: float | None = None,
    schedule_start_step: int | None = None,
    schedule_end_step: int | None = None,
) -> torch.Tensor:
    """Constraint on base height and orientation relative to terrain."""
    asset: Articulation = env.scene[asset_cfg.name]
    com_pos = asset.data.root_com_pos_w
    height = com_pos[:, 2]
    if terrain_sensor_cfg is not None:
        terrain_h = _terrain_height_at_points(
            env, terrain_sensor_cfg, com_pos.unsqueeze(1)
        ).squeeze(1)
        height = height - terrain_h
    if height_offset != 0.0:
        height = height - height_offset

    use_physical_schedule = any(
        value is not None
        for value in (
            height_range_start,
            height_range_end,
            max_angle_rad_start,
            max_angle_rad_end,
        )
    )
    if use_physical_schedule:
        alpha = _smoothstep_progress(env, schedule_start_step, schedule_end_step)
        start_range = height_range if height_range_start is None else height_range_start
        end_range = height_range if height_range_end is None else height_range_end
        min_h = _lerp(start_range[0], end_range[0], alpha)
        max_h = _lerp(start_range[1], end_range[1], alpha)
        start_ang = max_angle_rad if max_angle_rad_start is None else max_angle_rad_start
        end_ang = max_angle_rad if max_angle_rad_end is None else max_angle_rad_end
        max_angle = _lerp(start_ang, end_ang, alpha)
    else:
        min_h, max_h = height_range
        max_angle = max_angle_rad

    height_violation = (height < min_h) | (height > max_h)
    tilt = constraint_com_orientation(env, max_angle_rad=max_angle, asset_cfg=asset_cfg)
    cost = height_violation.float() + tilt
    scaled_limit = cost_limit
    if cost_limit is not None and not use_physical_schedule:
        phi = _dynamic_limit_scale(env, epsilon=limit_relax_epsilon, k=limit_relax_k)
        scaled_limit = float(cost_limit) * phi
    return _normalize_cost(cost, scaled_limit)


def gait_pattern_prob_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    gait_frequency: float = 1.0,
    phase_offsets: list[float] | None = None,
    stance_ratio: float = 0.5,
    contact_threshold: float = 1.0,
    command_name: str | None = None,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    max_command_speed: float | None = None,
    frequency_scale: float = 0.0,
    min_command_speed: float | None = None,
    min_base_speed: float | None = None,
    max_abs_yaw_cmd: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float | None = None,
    limit_relax_epsilon: float | None = None,
    limit_relax_k: float | None = None,
    gait_frequency_start: float | None = None,
    gait_frequency_end: float | None = None,
    stance_ratio_start: float | None = None,
    stance_ratio_end: float | None = None,
    phase_offset_scale_start: float | None = None,
    phase_offset_scale_end: float | None = None,
    phase_tolerance_start: float | None = None,
    phase_tolerance_end: float | None = None,
    schedule_start_step: int | None = None,
    schedule_end_step: int | None = None,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)

    use_physical_schedule = any(
        value is not None
        for value in (
            gait_frequency_start,
            gait_frequency_end,
            stance_ratio_start,
            stance_ratio_end,
            phase_offset_scale_start,
            phase_offset_scale_end,
            phase_tolerance_start,
            phase_tolerance_end,
        )
    )
    alpha = _smoothstep_progress(env, schedule_start_step, schedule_end_step)
    freq_start = gait_frequency if gait_frequency_start is None else gait_frequency_start
    freq_end = gait_frequency if gait_frequency_end is None else gait_frequency_end
    gait_frequency_curr = _lerp(freq_start, freq_end, alpha)
    stance_start = stance_ratio if stance_ratio_start is None else stance_ratio_start
    stance_end = stance_ratio if stance_ratio_end is None else stance_ratio_end
    stance_ratio_curr = _lerp(stance_start, stance_end, alpha)
    stance_ratio_curr = min(max(stance_ratio_curr, 0.05), 0.95)
    phase_scale_start = 1.0 if phase_offset_scale_start is None else phase_offset_scale_start
    phase_scale_end = 1.0 if phase_offset_scale_end is None else phase_offset_scale_end
    phase_offset_scale = _lerp(phase_scale_start, phase_scale_end, alpha)
    phase_tol_start = 0.0 if phase_tolerance_start is None else phase_tolerance_start
    phase_tol_end = 0.0 if phase_tolerance_end is None else phase_tolerance_end
    phase_tolerance = _lerp(phase_tol_start, phase_tol_end, alpha)
    phase_tolerance = min(max(phase_tolerance, 0.0), 0.49)

    dt = env.step_dt if hasattr(env, "step_dt") else 0.0
    phase_offsets = phase_offsets or [0.0] * len(foot_ids)
    # ensure phase list length matches number of feet
    phase_offsets = (phase_offsets + [0.0] * len(foot_ids))[: len(foot_ids)]
    phase_offsets = torch.as_tensor(
        phase_offsets, device=contact_sensor.data.net_forces_w.device, dtype=torch.float32
    )
    phase_offsets = phase_offsets * float(phase_offset_scale)
    freq = _resolve_gait_frequency(
        env,
        command_name=command_name,
        base_frequency=gait_frequency_curr,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        max_command_speed=max_command_speed,
        frequency_scale=frequency_scale,
        dtype=contact_sensor.data.net_forces_w.dtype,
        device=contact_sensor.data.net_forces_w.device,
    )
    t = _resolve_episode_phase_time(
        env,
        dt=dt,
        dtype=contact_sensor.data.net_forces_w.dtype,
        device=contact_sensor.data.net_forces_w.device,
    ).unsqueeze(-1)
    phase = (t * freq.unsqueeze(-1) + phase_offsets) % 1.0
    stance = phase < stance_ratio_curr

    # NOTE: expected layout is (N, H, B, 3); keep fallback for (N, B, H, 3).
    forces_hist = contact_sensor.data.net_forces_w_history
    max_foot_id = max(foot_ids)
    if forces_hist.ndim != 4:
        return _zeros_like_env(env)
    if forces_hist.shape[2] > max_foot_id:
        force_mag = forces_hist[:, :, foot_ids, :].norm(dim=-1).max(dim=1)[0]
    elif forces_hist.shape[1] > max_foot_id:
        force_mag = forces_hist[:, foot_ids, :, :].norm(dim=-1).max(dim=2)[0]
    else:
        return _zeros_like_env(env)
    contact = force_mag > contact_threshold

    mismatch = stance != contact
    if phase_tolerance > 0.0:
        dist_to_cycle_start = torch.minimum(phase, 1.0 - phase)
        dist_to_stance_edge = torch.abs(phase - stance_ratio_curr)
        transition_zone = (dist_to_cycle_start <= phase_tolerance) | (
            dist_to_stance_edge <= phase_tolerance
        )
        mismatch = mismatch & (~transition_zone)
    cost = mismatch.float().mean(dim=1)
    gate = _command_gate(
        env,
        command_name=command_name,
        min_command_speed=min_command_speed,
        min_base_speed=min_base_speed,
        asset_cfg=asset_cfg,
        device=cost.device,
        dtype=cost.dtype,
        max_abs_yaw_cmd=max_abs_yaw_cmd,
    )
    if gate is not None:
        cost = cost * gate
    scaled_limit = limit
    if limit is not None and not use_physical_schedule:
        phi = _dynamic_limit_scale(env, epsilon=limit_relax_epsilon, k=limit_relax_k)
        scaled_limit = float(limit) * phi
    return _normalize_cost(cost, scaled_limit)


def orthogonal_velocity_constraint(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    limit: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = getattr(asset.data, "root_lin_vel_b", None)
    if vel_b is not None:
        lateral_vel = vel_b[:, 1]
    else:
        lateral_vel = asset.data.root_lin_vel_w[:, 1]

    target_lateral = torch.zeros_like(lateral_vel)
    if command_name is not None and hasattr(env, "command_manager"):
        command = env.command_manager.get_command(command_name)
        if command is not None and command.shape[1] > 1:
            target_lateral = command[:, 1].to(
                device=lateral_vel.device, dtype=lateral_vel.dtype
            )

    cost = torch.abs(lateral_vel - target_lateral)
    return _normalize_cost(cost, limit)


def contact_velocity_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    contact_threshold: float = 1.0,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)
    forces = contact_sensor.data.net_forces_w_history
    contact_mask = forces.norm(dim=-1).max(dim=1)[0][:, foot_ids] > contact_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, foot_ids, :2]
    slip = torch.norm(foot_vel, dim=-1) * contact_mask.float()
    cost = slip.mean(dim=1)
    return _normalize_cost(cost, limit)


def foot_clearance_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    min_height: float | None = None,
    height_offset: float = 0.0,
    contact_threshold: float = 1.0,
    gait_frequency: float = 1.0,
    phase_offsets: list[float] | None = None,
    stance_ratio: float = 0.5,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    min_command_speed: float | None = None,
    min_base_speed: float | None = None,
    max_abs_yaw_cmd: float | None = None,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_heights = _foot_heights_relative(env, asset, foot_ids, terrain_sensor_cfg, height_offset)
    if min_height is None:
        min_height = 0.0
    min_height_t = torch.as_tensor(min_height, device=foot_heights.device, dtype=foot_heights.dtype)
    if min_height_t.ndim == 0:
        min_height_t = min_height_t.expand(foot_heights.shape[1])

    dt = env.step_dt if hasattr(env, "step_dt") else 0.0
    phase_offsets = phase_offsets or [0.0] * len(foot_ids)
    phase_offsets = phase_offsets[: len(foot_ids)]
    phase_offsets = torch.as_tensor(
        phase_offsets, device=foot_heights.device, dtype=foot_heights.dtype
    )
    freq = _resolve_gait_frequency(
        env,
        command_name=command_name,
        base_frequency=gait_frequency,
        min_frequency=None,
        max_frequency=None,
        max_command_speed=None,
        frequency_scale=0.0,
        dtype=foot_heights.dtype,
        device=foot_heights.device,
    )
    t = _resolve_episode_phase_time(
        env,
        dt=dt,
        dtype=foot_heights.dtype,
        device=foot_heights.device,
    ).unsqueeze(-1)
    phase = (t * freq.unsqueeze(-1) + phase_offsets) % 1.0
    swing = phase >= stance_ratio
    clearance = torch.clamp(min_height_t - foot_heights, min=0.0) * swing.float()
    cost = clearance.mean(dim=1)
    gate = _command_gate(
        env,
        command_name=command_name,
        min_command_speed=min_command_speed,
        min_base_speed=min_base_speed,
        asset_cfg=asset_cfg,
        device=cost.device,
        dtype=cost.dtype,
        max_abs_yaw_cmd=max_abs_yaw_cmd,
    )
    if gate is not None:
        cost = cost * gate
    return _normalize_cost(cost, limit)


def foot_height_limit_constraint(
    env: ManagerBasedRLEnv,
    foot_body_names: list[str],
    height_offset: float = 0.0,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)
    foot_heights = _foot_heights_relative(env, asset, foot_ids, terrain_sensor_cfg, height_offset)
    max_height = torch.max(foot_heights, dim=1).values
    if limit is None:
        return max_height
    limit_t = torch.as_tensor(limit, device=max_height.device, dtype=max_height.dtype)
    # Only penalize overshoot above the configured maximum height.
    cost = torch.clamp(max_height - limit_t, min=0.0)
    return _normalize_cost(cost, limit)


def symmetric_constraint(
    env: ManagerBasedRLEnv,
    joint_pair_indices: list[tuple[int, int]],
    action_pair_indices: list[tuple[int, int]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    include_actions: bool = True,
    command_name: str | None = None,
    min_command_speed: float | None = None,
    min_base_speed: float | None = None,
    max_abs_yaw_cmd: float | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Average action symmetry constraint using L1 distance on mirrored joints."""
    if not include_actions or not hasattr(env, "action_manager"):
        return _zeros_like_env(env)
    action_pairs = action_pair_indices or joint_pair_indices
    if not action_pairs:
        return _zeros_like_env(env)
    actions = env.action_manager.action
    sym = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    for left_idx, right_idx in action_pairs:
        sym += torch.abs(actions[:, left_idx] - actions[:, right_idx])
    sym /= len(action_pairs)
    gate = _command_gate(
        env,
        command_name=command_name,
        min_command_speed=min_command_speed,
        min_base_speed=min_base_speed,
        asset_cfg=asset_cfg,
        device=sym.device,
        dtype=sym.dtype,
        max_abs_yaw_cmd=max_abs_yaw_cmd,
    )
    if gate is not None:
        sym = sym * gate
    return _normalize_cost(sym, limit)


def base_contact_force_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str],
    threshold: float = 1.0,
    limit: float | None = None,
) -> torch.Tensor:
    """Continuous constraint on base contact force magnitude."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids, _ = contact_sensor.find_bodies(body_names, preserve_order=True)
    if not body_ids:
        return _zeros_like_env(env)
    net_forces = contact_sensor.data.net_forces_w_history
    force_mag = torch.norm(net_forces, dim=-1).max(dim=1)[0][:, body_ids]
    threshold_t = torch.as_tensor(threshold, device=force_mag.device, dtype=force_mag.dtype)
    violation = torch.clamp(force_mag - threshold_t, min=0.0)
    cost = torch.mean(violation, dim=1)
    return _normalize_cost(cost, limit)


def constraint_joint_vel(
    env: ManagerBasedRLEnv,
    limit: float = 50.0,
    soft_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft fraction of velocity-limit violations.

    ``soft_ratio`` controls where penalties start:
    - ``soft_ratio >= 1.0``: binary violation at ``|v| > limit`` (legacy behavior)
    - ``0 < soft_ratio < 1.0``: linear ramp from ``soft_ratio * limit`` to ``limit``
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_vel = getattr(asset.data, "joint_vel", None)
    if joint_vel is None:
        _warn_once(
            env,
            "_warn_missing_joint_vel",
            "Joint velocities not available; joint_vel constraint disabled.",
        )
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_vel = joint_vel[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=joint_vel.device, dtype=joint_vel.dtype)
    abs_vel = torch.abs(joint_vel)
    ratio = float(soft_ratio)
    if ratio >= 1.0:
        violation = abs_vel > limit_t
        return violation.float().mean(dim=1)

    ratio = min(max(ratio, 1.0e-6), 0.999999)
    soft_start = limit_t * ratio
    denom = torch.clamp(limit_t - soft_start, min=torch.finfo(joint_vel.dtype).eps)
    violation = torch.clamp((abs_vel - soft_start) / denom, min=0.0, max=1.0)
    return violation.mean(dim=1)


def constraint_joint_torque(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    soft_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft fraction of torque-limit violations.

    ``soft_ratio`` controls where penalties start:
    - ``soft_ratio >= 1.0``: binary violation at ``|tau| > limit`` (legacy behavior)
    - ``0 < soft_ratio < 1.0``: linear ramp from ``soft_ratio * limit`` to ``limit``
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    torque = getattr(asset.data, "applied_torque", None)
    if torque is None:
        _warn_once(
            env,
            "_warn_missing_applied_torque",
            "Applied torque not available; torque constraint disabled.",
        )
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        torque = torque[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=torque.device, dtype=torque.dtype)
    abs_torque = torch.abs(torque)
    ratio = float(soft_ratio)
    if ratio >= 1.0:
        violation = abs_torque > limit_t
        return violation.float().mean(dim=1)

    ratio = min(max(ratio, 1.0e-6), 0.999999)
    soft_start = limit_t * ratio
    denom = torch.clamp(limit_t - soft_start, min=torch.finfo(torque.dtype).eps)
    violation = torch.clamp((abs_torque - soft_start) / denom, min=0.0, max=1.0)
    return violation.mean(dim=1)


def constraint_com_orientation(
    env: ManagerBasedRLEnv,
    max_angle_rad: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary constraint for excessive base tilt."""
    projected_gravity = None
    try:
        from isaaclab.envs import mdp as isaaclab_mdp
    except ImportError:
        isaaclab_mdp = None
    if isaaclab_mdp is not None and hasattr(isaaclab_mdp, "projected_gravity"):
        projected_gravity = isaaclab_mdp.projected_gravity(env, asset_cfg)
    if projected_gravity is None:
        asset: Articulation = env.scene[asset_cfg.name]
        projected_gravity = getattr(asset.data, "projected_gravity_b", None)
    if projected_gravity is None:
        _warn_once(
            env,
            "_warn_missing_projected_gravity",
            "Projected gravity not available; tilt constraint disabled.",
        )
        return _zeros_like_env(env)
    gravity_xy = projected_gravity[:, :2]
    tilt = torch.norm(gravity_xy, dim=1)
    max_t = torch.as_tensor(max_angle_rad, device=tilt.device, dtype=tilt.dtype)
    return (tilt > math.sin(max_t)).float()


def _compute_contact_prob(
    contact_forces: torch.Tensor,
    threshold: float,
    dim: int = -1,
) -> torch.Tensor:
    threshold_t = torch.as_tensor(
        threshold, device=contact_forces.device, dtype=contact_forces.dtype
    )
    magnitude = torch.norm(contact_forces, dim=dim)
    return (magnitude > threshold_t).float()


def _get_contact_prob(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str],
    threshold: float,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids, _ = contact_sensor.find_bodies(body_names, preserve_order=True)
    if not body_ids:
        return _zeros_like_env(env)
    forces = contact_sensor.data.net_forces_w_history
    forces = forces[:, body_ids]
    prob = _compute_contact_prob(forces, threshold)
    return prob.mean(dim=1)
