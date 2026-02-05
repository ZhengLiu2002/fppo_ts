# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedRLEnv

def time_out(
    env: CRLManagerBasedRLEnv,
):  
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_limit: float = 1.5,
    pitch_limit: float = 1.5,
):
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:, 3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > roll_limit
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > pitch_limit
    return roll_cutoff | pitch_cutoff


def base_height_below(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = -0.25,
):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_state_w[:, 2] < height_threshold


def body_contact(
    env: CRLManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
):
    """Terminate when non-foot body contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


def terrain_out_of_bounds(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the actor moves too close to the edge of the terrain."""
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False
    elif env.scene.cfg.terrain.terrain_type == "generator":
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width
        asset: Articulation = env.scene[asset_cfg.name]
        x_out = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out, y_out)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
