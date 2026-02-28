from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedEnv
    from .crl_command_cfg import CRLCommandCfg


class UniformCRLCommand(CommandTerm):
    cfg: CRLCommandCfg

    def __init__(self, cfg: CRLCommandCfg, env: CRLManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Import GalileoDefaults to access terrain-specific command ranges.
        try:
            from crl_tasks.tasks.galileo.config.defaults import GalileoDefaults

            self.terrain_ranges = GalileoDefaults.command.ranges
        except ImportError:
            omni.log.warn("Cannot import GalileoDefaults, falling back to default ranges")
            self.terrain_ranges = None

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}\n"
        msg += f"\tSmall command to zero: {self.cfg.small_commands_to_zero}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _get_terrain_name(self, env_ids: Sequence[int]) -> str | None:
        """获取指定环境的地形名称。"""
        if not hasattr(self._env.scene, "terrain"):
            return None

        terrain = self._env.scene.terrain
        # 尝试从 crl_event 获取地形名称
        if hasattr(self._env, "crl_manager") and self._env.crl_manager is not None:
            try:
                crl_event = self._env.crl_manager._terms.get("base_crl")
                if crl_event is not None and hasattr(crl_event, "env_per_terrain_name"):
                    terrain_names = crl_event.env_per_terrain_name
                    if len(env_ids) > 0:
                        # 获取第一个环境的地形名称（假设同一批次环境在同一地形）
                        first_env_id = (
                            env_ids[0] if isinstance(env_ids, (list, tuple)) else env_ids[0].item()
                        )
                        if first_env_id < len(terrain_names):
                            return str(terrain_names[first_env_id])
            except Exception:
                pass

        # 如果无法从 crl_event 获取，尝试直接从 terrain 获取
        if hasattr(terrain, "terrain_generator_class"):
            terrain_gen = terrain.terrain_generator_class
            if (
                hasattr(terrain_gen, "terrain_names")
                and hasattr(terrain, "terrain_levels")
                and hasattr(terrain, "terrain_types")
            ):
                try:
                    first_env_id = (
                        env_ids[0] if isinstance(env_ids, (list, tuple)) else env_ids[0].item()
                    )
                    level = terrain.terrain_levels[first_env_id].item()
                    terrain_type = terrain.terrain_types[first_env_id].item()
                    terrain_names = terrain_gen.terrain_names
                    if level < terrain_names.shape[0] and terrain_type < terrain_names.shape[1]:
                        return str(terrain_names[level, terrain_type])
                except Exception:
                    pass

        return None

    def _get_terrain_specific_ranges(self, terrain_name: str | None):
        """根据地形名称获取对应的指令范围配置。

        如果找不到对应的配置，会记录警告并使用 None，强制使用 cfg.ranges 中的默认值。
        这样可以确保每种地形都必须有明确的配置，避免静默使用错误的配置。
        """
        if terrain_name is None or self.terrain_ranges is None:
            return None

        # 如果地形名称在配置中存在，返回对应的配置
        if terrain_name in self.terrain_ranges:
            return self.terrain_ranges[terrain_name]

        # 如果找不到配置，记录警告（但不在每次调用时都打印，避免日志过多）
        if not hasattr(self, "_terrain_warning_printed"):
            self._terrain_warning_printed = set()

        if terrain_name not in self._terrain_warning_printed:
            omni.log.warn(
                f"[UniformCRLCommand] 未找到地形 '{terrain_name}' 的指令配置，"
                f"将使用 cfg.ranges 中的默认配置。请确保在 GalileoDefaults.command.ranges 中为所有地形添加配置。"
            )
            self._terrain_warning_printed.add(terrain_name)

        # 返回 None，强制使用 cfg.ranges 中的默认值
        return None

    def _resample_command(self, env_ids: Sequence[int]):
        # 获取地形名称并选择对应的指令配置
        terrain_name = self._get_terrain_name(env_ids)
        terrain_ranges = self._get_terrain_specific_ranges(terrain_name)
        # If command-threshold curriculum is active, those terms update cfg.ranges.lin_vel_x/ang_vel_z.
        # Keep these as the single source instead of overriding with terrain-specific static ranges.
        use_lin_x_cfg_range = (
            self.cfg.ranges.start_curriculum_lin_x is not None
            and self.cfg.ranges.max_curriculum_lin_x is not None
        )
        use_ang_z_cfg_range = (
            self.cfg.ranges.start_curriculum_ang_z is not None
            and self.cfg.ranges.max_curriculum_ang_z is not None
        )

        # 如果找到地形特定的配置，使用它；否则使用默认配置
        if terrain_ranges is not None:
            # 使用地形特定的配置
            lin_vel_x_base = (
                self.cfg.ranges.lin_vel_x if use_lin_x_cfg_range else terrain_ranges["lin_vel_x"]
            )
            lin_vel_y_base = terrain_ranges["lin_vel_y"]
            ang_vel_z_base = (
                self.cfg.ranges.ang_vel_z if use_ang_z_cfg_range else terrain_ranges["ang_vel_z"]
            )
            heading_base = terrain_ranges.get("heading", self.cfg.ranges.heading)
            heading_command_prob = terrain_ranges.get(
                "heading_command_prob", self.cfg.ranges.heading_command_prob
            )
            yaw_command_prob = terrain_ranges.get(
                "yaw_command_prob", self.cfg.ranges.yaw_command_prob
            )
            standing_command_prob = terrain_ranges.get(
                "standing_command_prob", self.cfg.ranges.standing_command_prob
            )
            start_curriculum_lin_x = terrain_ranges.get(
                "start_curriculum_lin_x", self.cfg.ranges.start_curriculum_lin_x
            )
            start_curriculum_ang_z = terrain_ranges.get(
                "start_curriculum_ang_z", self.cfg.ranges.start_curriculum_ang_z
            )
            max_curriculum_lin_x = terrain_ranges.get(
                "max_curriculum_lin_x", self.cfg.ranges.max_curriculum_lin_x
            )
            max_curriculum_ang_z = terrain_ranges.get(
                "max_curriculum_ang_z", self.cfg.ranges.max_curriculum_ang_z
            )
        else:
            # 使用默认配置
            lin_vel_x_base = self.cfg.ranges.lin_vel_x
            lin_vel_y_base = self.cfg.ranges.lin_vel_y
            ang_vel_z_base = self.cfg.ranges.ang_vel_z
            heading_base = self.cfg.ranges.heading
            heading_command_prob = self.cfg.ranges.heading_command_prob
            yaw_command_prob = self.cfg.ranges.yaw_command_prob
            standing_command_prob = self.cfg.ranges.standing_command_prob
            start_curriculum_lin_x = self.cfg.ranges.start_curriculum_lin_x
            start_curriculum_ang_z = self.cfg.ranges.start_curriculum_ang_z
            max_curriculum_lin_x = self.cfg.ranges.max_curriculum_lin_x
            max_curriculum_ang_z = self.cfg.ranges.max_curriculum_ang_z

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        lin_x_range = lin_vel_x_base
        ang_z_range = ang_vel_z_base
        progress = None
        # curriculum-aware range scaling (optional)
        if (
            self.cfg.terrain_level_range_scaling
            and self.cfg.ranges.start_curriculum_lin_x is not None
            and self.cfg.ranges.max_curriculum_lin_x is not None
            and hasattr(self._env.scene, "terrain")
            and getattr(self._env.scene.terrain, "terrain_levels", None) is not None
        ):
            levels = self._env.scene.terrain.terrain_levels[env_ids].float()
            terrain_gen = getattr(self._env.scene.terrain, "terrain_generator_class", None)
            num_rows = float(getattr(terrain_gen, "num_rows", 1))
            denom = max(num_rows - 1.0, 1.0)
            progress = torch.clamp(levels / denom, 0.0, 1.0)
            start_min, start_max = (
                start_curriculum_lin_x if start_curriculum_lin_x is not None else lin_vel_x_base
            )
            max_min, max_max = (
                max_curriculum_lin_x if max_curriculum_lin_x is not None else lin_vel_x_base
            )
            lin_x_range = (
                float(start_min + (max_min - start_min) * progress.mean().item()),
                float(start_max + (max_max - start_max) * progress.mean().item()),
            )
        if (
            progress is not None
            and start_curriculum_ang_z is not None
            and max_curriculum_ang_z is not None
        ):
            start_min, start_max = start_curriculum_ang_z
            max_min, max_max = max_curriculum_ang_z
            ang_z_range = (
                float(start_min + (max_min - start_min) * progress.mean().item()),
                float(start_max + (max_max - start_max) * progress.mean().item()),
            )
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*lin_x_range)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*lin_vel_y_base)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*ang_z_range)
        # heading target
        if self.cfg.heading_command and heading_base is not None:
            self.heading_target[env_ids] = r.uniform_(*heading_base)
            # 使用地形特定的 heading_command_prob，如果存在；否则使用配置中的值
            heading_prob = (
                heading_command_prob if terrain_ranges is not None else self.cfg.rel_heading_envs
            )
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= heading_prob
        # update standing envs
        # 使用地形特定的 standing_command_prob，如果存在；否则使用配置中的值
        standing_prob = (
            standing_command_prob if terrain_ranges is not None else self.cfg.rel_standing_envs
        )
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= standing_prob
        # update standing envs
        if self.cfg.small_commands_to_zero:
            self.vel_command_b[env_ids, :2] *= (
                torch.abs(self.vel_command_b[env_ids, 0:1]) > self.cfg.clips.lin_vel_clip
            )

        # enforce minimum absolute command magnitudes (optional)
        min_abs_x = self.cfg.min_abs_lin_vel_x
        if min_abs_x is not None and min_abs_x > 0.0:
            x_vals = self.vel_command_b[env_ids, 0]
            x_abs = torch.abs(x_vals)
            mask = x_abs < min_abs_x
            if mask.any():
                signs = torch.sign(x_vals[mask])
                zeros = signs == 0
                if zeros.any():
                    signs[zeros] = torch.sign(torch.rand_like(signs[zeros]) - 0.5)
                self.vel_command_b[env_ids[mask], 0] = signs * min_abs_x

        min_abs_y = self.cfg.min_abs_lin_vel_y
        if min_abs_y is not None and min_abs_y > 0.0:
            y_vals = self.vel_command_b[env_ids, 1]
            y_abs = torch.abs(y_vals)
            mask = y_abs < min_abs_y
            if mask.any():
                signs = torch.sign(y_vals[mask])
                zeros = signs == 0
                if zeros.any():
                    signs[zeros] = torch.sign(torch.rand_like(signs[zeros]) - 0.5)
                self.vel_command_b[env_ids[mask], 1] = signs * min_abs_y

    def _update_command(self):
        # Optional alignment to crl goals for legacy tasks
        if self.cfg.align_to_crl_goal and hasattr(self._env, "crl_manager"):
            crl_term = self._env.crl_manager._terms.get("base_crl")
            if crl_term and hasattr(crl_term, "target_yaw"):
                self.heading_target = crl_term.target_yaw
                self.is_heading_env[:] = True

        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
                )
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # -- Goal Arrow (Green): Points towards the target yaw (heading_target)
        # Create arrow quaternion directly from heading_target (World Frame)
        zeros = torch.zeros_like(self.heading_target)
        vel_des_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_target)

        # Scale based on commanded linear velocity magnitude
        cmd_mag = torch.norm(self.command[:, :2], dim=1)
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        vel_des_arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            self.num_envs, 1
        )
        vel_des_arrow_scale[:, 0] *= cmd_mag * 3.0

        # -- Current Velocity Arrow (Blue): Points in direction of actual movement
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )

        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
