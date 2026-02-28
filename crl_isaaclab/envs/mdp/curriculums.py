# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for CRL tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    episodes_per_level: int | None = None,
) -> torch.Tensor:
    """Update terrain levels based on distance walked under velocity commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # 更宽松的晋级阈值，鼓励更快推进地形关卡
    move_up = distance > terrain.cfg.terrain_generator.size[0] * 0.25
    # 放宽降级条件，避免频繁回退
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.3
    move_down *= ~move_up
    if episodes_per_level is not None and episodes_per_level > 0:
        max_episode_length = env.max_episode_length
        min_steps = (
            (terrain.terrain_levels[env_ids].float() + 1.0)
            * max_episode_length
            * episodes_per_level
        )
        move_up &= env.common_step_counter > min_steps

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_x_command_threshold(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], episodes_per_level: int = 8
) -> torch.Tensor:
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length
    lin_x_level = float(getattr(command.cfg, "lin_x_level", 0.0))
    max_lin_x_level = float(getattr(command.cfg, "max_lin_x_level", 1.0))
    lin_x_level_step = float(getattr(command.cfg, "lin_x_level_step", 1.0))
    lin_x_level_step = max(lin_x_level_step, 0.0)

    lin_x_level = max(0.0, min(lin_x_level, max_lin_x_level))
    if (
        env.common_step_counter > ((lin_x_level + 1.0) * max_episode_length * episodes_per_level)
    ) and (lin_x_level < max_lin_x_level):
        if lin_x_level_step > 0.0:
            lin_x_level = min(lin_x_level + lin_x_level_step, max_lin_x_level)
        else:
            lin_x_level = max_lin_x_level
        command.cfg.lin_x_level = lin_x_level

    # Always apply the current curriculum level to the command ranges (keeps cfg consistent
    # even when the level hasn't changed yet).
    denom = max(float(max_lin_x_level), 1.0e-6)
    ranges = command.cfg.ranges
    if (
        hasattr(ranges, "start_curriculum_lin_x")
        and ranges.start_curriculum_lin_x is not None
        and hasattr(ranges, "max_curriculum_lin_x")
        and ranges.max_curriculum_lin_x is not None
    ):
        start_min, start_max = ranges.start_curriculum_lin_x
        max_min, max_max = ranges.max_curriculum_lin_x
        step0 = (max_min - start_min) / denom
        step1 = (max_max - start_max) / denom
        ranges.lin_vel_x = (start_min + step0 * lin_x_level, start_max + step1 * lin_x_level)

    return torch.tensor(lin_x_level, device=env.device)


def ang_vel_z_command_threshold(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], episodes_per_level: int = 8
) -> torch.Tensor:
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length
    ang_z_level = float(getattr(command.cfg, "ang_z_level", 0.0))
    max_ang_z_level = float(getattr(command.cfg, "max_ang_z_level", 1.0))
    ang_z_level_step = float(getattr(command.cfg, "ang_z_level_step", 1.0))
    ang_z_level_step = max(ang_z_level_step, 0.0)

    ang_z_level = max(0.0, min(ang_z_level, max_ang_z_level))
    if (
        env.common_step_counter > ((ang_z_level + 1.0) * max_episode_length * episodes_per_level)
    ) and (ang_z_level < max_ang_z_level):
        if ang_z_level_step > 0.0:
            ang_z_level = min(ang_z_level + ang_z_level_step, max_ang_z_level)
        else:
            ang_z_level = max_ang_z_level
        command.cfg.ang_z_level = ang_z_level

    # Always apply the current curriculum level to the command ranges.
    denom = max(float(max_ang_z_level), 1.0e-6)
    ranges = command.cfg.ranges
    if (
        hasattr(ranges, "start_curriculum_ang_z")
        and ranges.start_curriculum_ang_z is not None
        and hasattr(ranges, "max_curriculum_ang_z")
        and ranges.max_curriculum_ang_z is not None
    ):
        start_min, start_max = ranges.start_curriculum_ang_z
        max_min, max_max = ranges.max_curriculum_ang_z
        step0 = (max_min - start_min) / denom
        step1 = (max_max - start_max) / denom
        ranges.ang_vel_z = (start_min + step0 * ang_z_level, start_max + step1 * ang_z_level)

    return torch.tensor(ang_z_level, device=env.device)
