
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np 
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import wrap_to_pi
from crl_isaaclab.managers import CRLTerm
from crl_isaaclab.terrains import CRLTerrainGeneratorCfg, CRLTerrainImporter, CRLTerrainGenerator

if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedRLEnv
    from .crl_events_cfg import CRLEventsCfg


class CRLEvent(CRLTerm):
    cfg: CRLEventsCfg

    def __init__(
        self, 
        cfg: CRLEventsCfg, 
        env: CRLManagerBasedRLEnv
        ):
        super().__init__(cfg, env)

        self.episode_length_s = env.cfg.episode_length_s
        self.reach_goal_delay = cfg.reach_goal_delay
        self.num_future_goal_obs = cfg.num_future_goal_obs
        self.next_goal_threshold = cfg.next_goal_threshold
        self.simulation_time = env.step_dt
        self.arrow_num = cfg.arrow_num
        self.env = env 
        self.debug_vis = cfg.debug_vis
        self.promotion_goal_threshold = cfg.promotion_goal_threshold
        self.demotion_goal_threshold = cfg.demotion_goal_threshold
        self.promotion_distance_ratio = cfg.promotion_distance_ratio
        self.demotion_distance_ratio = cfg.demotion_distance_ratio
        self.distance_progress_cap = cfg.distance_progress_cap
               
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- metrics
        self.metrics["far_from_current_goal"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["how_far_from_start_point"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["terrain_levels"] = torch.zeros(self.num_envs, device='cpu')
        self.metrics["current_goal_idx"] = torch.zeros(self.num_envs, device='cpu')
        self.dis_to_start_pos = torch.zeros(self.num_envs, device=self.device)
        self.terrain: CRLTerrainImporter = self.env.scene.terrain
        terrain_generator: CRLTerrainGenerator = self.terrain.terrain_generator_class
        crl_terrain_cfg :CRLTerrainGeneratorCfg = self.terrain.cfg.terrain_generator
        # stage-based curriculum state (P0~P4)
        self.stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_attempts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stage_eval_window = 8
        self.stage_min_stay = 6
        self.max_stage = 100  # allow unbounded progression
        self.recall_prob = 0.15
        # linear mapping stage->terrain row (spacing 2), will be clamped by available rows
        self.stage_to_level = torch.arange(0, 200, 2, device=self.device, dtype=torch.long)
        self.reached_goal_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # expose stage for other components (e.g., hurdle placement)
        self.env.curriculum_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.num_goals = crl_terrain_cfg.num_goals
        self.env_class = torch.zeros(self.num_envs, device=self.device)
        self.env_origins = self.terrain.env_origins
        self.terrain_type = terrain_generator.terrain_type
        self.terrain_class = torch.from_numpy(self.terrain_type).to(self.device).to(torch.float)
        self.env_class[:] = self.terrain_class[self.terrain.terrain_levels, self.terrain.terrain_types]
        
        terrain_goals = terrain_generator.goals
        self.terrain_goals = torch.from_numpy(terrain_goals).to(self.device).to(torch.float)
        self.env_goals = torch.zeros(
            self.num_envs,
            self.terrain_goals.shape[2] + self.num_future_goal_obs,
            3,
            device=self.device,
            requires_grad=False,
        )
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._refresh_env_goals()
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float).to(device = self.device)
        
        if self.debug_vis:
            self.total_heights = torch.from_numpy(terrain_generator.goal_heights).to(device = self.device)
            self.future_goal_idx = torch.ones(self.num_goals, device=self.device, dtype=torch.bool).repeat(self.num_envs, 1)
            self.future_goal_idx[:, 0] = False
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]
       
        self.total_terrain_names = terrain_generator.terrain_names
        numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy() ## string type can't convert to torch
        numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
        self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]
        self._reset_offset = self.env.event_manager.get_term_cfg('reset_root_state').params['offset']

        self._update_goal_vectors()


    def __call__(self):
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self._update_goal_vectors()

    def _gather_cur_goals(self, future=0):
        idx = torch.clamp(self.cur_goal_idx + future, max=self.env_goals.shape[1] - 1)
        return self.env_goals.gather(1, idx[:, None, None].expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def __str__(self) -> str:
        msg = "CRLCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg
    
    def _update_command(self):
        """Re-target the current goal position to the current root state."""
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.reached_goal_ids = torch.norm(robot_root_pos_w - self.cur_goals[:, :2], dim=1) < self.next_goal_threshold
        reached_goal_idx = self.reached_goal_ids.nonzero(as_tuple=False).squeeze(-1)
        if reached_goal_idx.numel() > 0:
            self.reach_goal_timer[reached_goal_idx] += 1
        # require连续驻留才能切换目标，离开判定区域即清零计时
        self.reach_goal_timer[~self.reached_goal_ids] = 0
        
        # 更新可视化状态：在机器人到达目标点时立即更新，让目标点立即变绿
        if self.debug_vis and self.reached_goal_ids.any():
            # 只更新未完成的目标（避免索引越界）
            valid_mask = (self.cur_goal_idx >= 0) & (self.cur_goal_idx < self.num_goals) & self.reached_goal_ids
            if valid_mask.any():
                # 标记当前目标为已到达（变绿）
                self.future_goal_idx[valid_mask, self.cur_goal_idx[valid_mask]] = False
        
        # 检查是否可以切换到下一个目标：需要满足驻留时间要求且当前目标索引未完成
        next_flag = self.reach_goal_timer > self.reach_goal_delay / self.simulation_time
        # 避免索引越界，仅对未完成的索引做自增
        next_flag = next_flag & (self.cur_goal_idx < self.num_goals - 1)  # 修改：最后一个目标需要特殊处理
        
        # 切换到下一个目标
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0
        
        # 处理最后一个目标：到达最后一个目标后，需要持续驻留才能完成
        # 只有当到达最后一个目标（索引为 num_goals - 1）且满足驻留时间时，才标记为完成
        last_goal_mask = (self.cur_goal_idx == (self.num_goals - 1)) & self.reached_goal_ids
        last_goal_timer_ok = self.reach_goal_timer > self.reach_goal_delay / self.simulation_time
        last_goal_complete = last_goal_mask & last_goal_timer_ok
        if last_goal_complete.any():
            if self.debug_vis:
                # 标记最后一个目标为已到达
                self.future_goal_idx[last_goal_complete, self.num_goals - 1] = False
            # 标记为完成，触发终止条件
            self.cur_goal_idx[last_goal_complete] = self.num_goals

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self._update_goal_vectors(robot_root_pos_w)
        start_pos = self.env_origins[:,:2] - \
                    torch.tensor((self.terrain.cfg.terrain_generator.size[1] + \
                                  self._reset_offset, 0)).to(self.device)

        self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[:, :2], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # 如果关闭课程（如 Play），跳过阶段机，但仍需重置目标/计时器
        curriculum_on = bool(getattr(self.terrain.cfg.terrain_generator, "curriculum", True))

        if curriculum_on:
            # record outcome of last episode for these envs
            success_mask, has_outcome = self._get_last_outcome(env_ids)
            self._update_stage(env_ids, success_mask, has_outcome)

            # curriculum recall: small prob to revisit previous stage for retention
            stage_now = self.stage.clone()
            stage_used = stage_now[env_ids]
            recall_mask = (stage_used > 0) & (torch.rand_like(stage_used.float()) < self.recall_prob)
            stage_used = torch.where(recall_mask, stage_used - 1, stage_used)
            self.env.curriculum_stage[env_ids] = stage_used

            # map stage to terrain level row; clamp by available rows
            max_rows = self.terrain.terrain_origins.shape[0]
            level_values = self.stage_to_level[torch.clamp(stage_used, max=self.stage_to_level.shape[0] - 1)]
            level_values = torch.clamp(level_values, max=max_rows - 1)
            self.terrain.terrain_levels[env_ids] = level_values
            self.env_origins[env_ids] = self.terrain.terrain_origins[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
            self.env_class[env_ids] = self.terrain_class[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]

        self._refresh_env_goals(env_ids)
        self._update_goal_vectors()

        if curriculum_on:
            numpy_terrain_levels = self.terrain.terrain_levels.detach().cpu().numpy()
            numpy_terrain_types = self.terrain.terrain_types.detach().cpu().numpy()
            self.env_per_terrain_name = self.total_terrain_names[numpy_terrain_levels, numpy_terrain_types]

        # 重置目标相关状态：确保完全重置，避免残留状态导致问题
        self.reach_goal_timer[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0
        self.reached_goal_ids[env_ids] = False
        
        # 更新当前目标位置（重置后应该指向第一个目标）
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        if self.debug_vis:
            # 重置可视化状态：第一个目标标记为当前（已到达），其他标记为未来（未到达）
            self.future_goal_idx[env_ids, 0] = False
            self.future_goal_idx[env_ids, 1:] = True
            self.env_per_heights = self.total_heights[self.terrain.terrain_levels, self.terrain.terrain_types]

    def _update_metrics(self):
        # logs data
        levels = self.terrain.terrain_levels.float()
        self.metrics["terrain_levels"] = levels.to(device='cpu')
        self.metrics["terrain_level_mean"] = torch.mean(levels).to(device='cpu')
        self.metrics["terrain_level_max"] = torch.max(levels).to(device='cpu')
        self.metrics["curriculum_stage"] = self.stage.to(device="cpu", dtype=torch.float)
        robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        self.metrics["far_from_current_goal"] = (torch.norm(self.cur_goals[:, :2] - robot_root_pos_w,dim =-1) - self.next_goal_threshold).to(device = 'cpu')
        self.metrics["current_goal_idx"] = self.cur_goal_idx.to(device='cpu', dtype=float)
        self.metrics["how_far_from_start_point"] = self.dis_to_start_pos.to(device = 'cpu')

    def _refresh_env_goals(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Gather and pad terrain goals, then recentre and offset them safely away from hurdles."""
        if env_ids is not None:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        temp = self.terrain_goals[self.terrain.terrain_levels, self.terrain.terrain_types]
        temp = self._apply_hurdle_layout_goals(temp, env_ids)
        last_col = temp[:, -1].unsqueeze(1)
        padded = torch.cat((temp, last_col.repeat(1, self.num_future_goal_obs, 1)), dim=1)
        # 将所有目标点对齐到赛道中心线，避免沿 y 轴随机漂移
        padded[:, :, 1] = 0.0
        padded = self._offset_goals_from_hurdles(padded)
        if env_ids is None:
            self.env_goals.copy_(padded)
        else:
            self.env_goals[env_ids] = padded[env_ids]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _offset_goals_from_hurdles(self, goal_tensor: torch.Tensor) -> torch.Tensor:
        """Push goals away from hurdles so that the policy never aims directly at an obstacle."""
        try:
            place_cfg = self.env.event_manager.get_term_cfg("place_hurdles")
            spacing = float(place_cfg.params.get("spacing", 2.0))
            start = float(place_cfg.params.get("start", 2.0))
            hurdle_count = int(place_cfg.params.get("count", 4))
        except Exception:
            return goal_tensor

        if hurdle_count <= 0:
            return goal_tensor

        hurdle_slots = start + spacing * torch.arange(hurdle_count, device=self.device)
        env_hurdle_x = hurdle_slots.view(1, -1).expand(goal_tensor.shape[0], -1)
        goal_x = goal_tensor[:, :, 0]
        margin = float(place_cfg.params.get("goal_margin", 0.5))
        dist = goal_x.unsqueeze(-1) - env_hurdle_x.unsqueeze(1)
        nearest = torch.abs(dist).min(dim=-1).values
        near_mask = nearest < margin
        if not near_mask.any():
            return goal_tensor

        nearest_idx = torch.abs(dist).argmin(dim=-1)
        nearest_hurdle_x = env_hurdle_x.gather(1, nearest_idx)
        adjusted_x = torch.where(goal_x < nearest_hurdle_x, nearest_hurdle_x + margin, nearest_hurdle_x - margin)
        goal_tensor[:, :, 0] = torch.where(near_mask, adjusted_x, goal_x)
        return goal_tensor

    def _apply_hurdle_layout_goals(
        self, base_goals: torch.Tensor, env_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Project waypoints to the Galileo hurdle layout (entry + per-hurdle + final, then padded).
        
        布局说明：
        - 目标 0: entry point (起点前)
        - 目标 1 到 1+visible-1: 每个可见 hurdle 的位置
        - 目标 num_goals-1: final goal (最后一个 hurdle 之后)
        
        注意：num_goals 应该等于 1 + hurdle_count + 1 (entry + hurdles + final)
        例如：4 个 hurdles 需要 num_goals = 6
        """
        try:
            place_cfg = self.env.event_manager.get_term_cfg("place_hurdles")
        except Exception:
            return base_goals

        hurdle_count = int(place_cfg.params.get("count", 4))
        if hurdle_count <= 0:
            return base_goals

        spacing = float(place_cfg.params.get("spacing", 2.0))
        start = float(place_cfg.params.get("start", 2.0))
        entry_offset = float(place_cfg.params.get("entry_offset", spacing * 0.5))
        post_goal_offset = float(place_cfg.params.get("post_goal_offset", spacing * 0.75))
        layout = place_cfg.params.get("layout", "auto")
        curriculum_on = getattr(self.terrain.cfg.terrain_generator, "curriculum", True)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if layout != "auto" or not curriculum_on:
            visible = torch.full((env_ids.numel(),), hurdle_count, device=self.device, dtype=torch.long)
        else:
            stage_tensor = getattr(self.env, "curriculum_stage", None)
            if stage_tensor is not None:
                stage_used = torch.clamp(stage_tensor[env_ids], 0, 50)
            else:
                stage_used = torch.clamp(self.terrain.terrain_levels[env_ids] // 2, 0, 50)
            unlock_interval = max(int(place_cfg.params.get("warmup_levels", 1)), 1)
            visible = torch.clamp(stage_used // unlock_interval + 1, max=hurdle_count)

        size_x = float(self.terrain.cfg.terrain_generator.size[0])
        # keep navigation points away from terrain borders to avoid falling off the mesh
        margin = 0.5
        min_x = -0.5 * size_x + margin
        max_x = 0.5 * size_x - margin

        entry_x = torch.clamp(
            torch.full((env_ids.numel(),), start - entry_offset, device=self.device), min_x, max_x
        )
        last_x = start + spacing * torch.clamp(visible.to(torch.float) - 1.0, min=0.0)
        final_x = torch.clamp(last_x + post_goal_offset, min_x, max_x)

        slot_max = min(hurdle_count, max(self.num_goals - 2, 0))
        base_x = final_x.unsqueeze(1).repeat(1, self.num_goals)
        base_x[:, 0] = entry_x
        if slot_max > 0:
            hurdle_xs = torch.clamp(
                start + spacing * torch.arange(slot_max, device=self.device, dtype=torch.float), min_x, max_x
            )
            mask = torch.arange(slot_max, device=self.device, dtype=torch.long).unsqueeze(0) < visible.unsqueeze(1)
            hurdle_grid = hurdle_xs.unsqueeze(0).expand(env_ids.numel(), slot_max)
            base_slice = base_x[:, 1 : 1 + slot_max]
            base_x[:, 1 : 1 + slot_max] = torch.where(mask, hurdle_grid, base_slice)

        updated_goals = base_goals.clone()
        updated_goals[env_ids, : self.num_goals, 0] = base_x
        updated_goals[env_ids, : self.num_goals, 1] = 0.0
        return updated_goals

    def _update_goal_vectors(self, robot_root_pos_w: torch.Tensor | None = None):
        """Recompute goal-relative vectors and yaws for the latest goal indices."""
        if robot_root_pos_w is None:
            robot_root_pos_w = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]

        self.target_pos_rel = self.cur_goals[:, :2] - robot_root_pos_w
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_root_pos_w

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def _get_last_outcome(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode success per env with a conservative definition."""
        if not hasattr(self.env, "reset_time_outs") or self.env.reset_time_outs.numel() == 0:
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool), torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        try:
            timeout_mask = self.env.reset_time_outs[env_ids].to(torch.bool)
        except Exception:
            return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool), torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        reached_goal = self.reached_goal_ids[env_ids] if hasattr(self, "reached_goal_ids") else torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
        # 移除距离判定：必须到达目标点才算成功，避免“跑过半程”导致的伪成功
        # start_pos = self.env_origins[env_ids, :2] - torch.tensor((self.terrain.cfg.terrain_generator.size[1] + self._reset_offset, 0)).to(self.device)
        # dist = torch.norm(start_pos - self.robot.data.root_pos_w[env_ids, :2], dim=1)
        # dist_thresh = 0.5 * (self.terrain.cfg.terrain_generator.size[1])
        # dist_success = dist > dist_thresh
        success = reached_goal
        # 超时不再视为成功，避免“站桩升级”
        return success, torch.ones(len(env_ids), device=self.device, dtype=torch.bool)

    def _update_stage(self, env_ids: Sequence[int], success_mask: torch.Tensor, valid_mask: torch.Tensor):
        """Stage state machine: accumulate windowed success rate; promote/demote after cooldown."""
        if len(env_ids) == 0:
            return
        # 对无效的样本跳过计数
        if not torch.any(valid_mask):
            return
        # update counters
        self.stage_attempts[env_ids] += valid_mask.to(torch.long)
        self.stage_success[env_ids] += (success_mask & valid_mask).to(torch.long)
        self.stage_cooldown[env_ids] += 1

        # evaluate when enough attempts and cooldown satisfied
        eval_mask = (self.stage_attempts >= self.stage_eval_window) & (self.stage_cooldown >= self.stage_min_stay)
        success_rate = torch.where(self.stage_attempts > 0, self.stage_success.float() / self.stage_attempts.float(), torch.zeros_like(self.stage_success, dtype=torch.float))

        promote = eval_mask & (success_rate >= 0.7) & (self.stage < self.max_stage)
        demote = eval_mask & (success_rate <= 0.3) & (self.stage > 0)

        self.stage = torch.clamp(self.stage + promote.to(torch.long) - demote.to(torch.long), 0, self.max_stage)

        # reset counters for envs that changed or were evaluated
        reset_mask = promote | demote
        self.stage_success[reset_mask] = 0
        self.stage_attempts[reset_mask] = 0
        self.stage_cooldown[reset_mask] = 0

        # avoid counter explosion; keep attempts bounded
        self.stage_attempts = torch.clamp(self.stage_attempts, max=self.stage_eval_window * 2)
        self.stage_success = torch.clamp(self.stage_success, max=self.stage_eval_window * 2)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer = VisualizationMarkers(self.cfg.current_goal_pose_visualizer_cfg)
            # set their visibility to true
            self.current_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer = VisualizationMarkers(self.cfg.future_goal_poses_visualizer_cfg)
            self.future_goal_poses_visualizer.set_visibility(True)


            if not hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer = VisualizationMarkers(self.cfg.current_arrow_visualizer_cfg)
            # set their visibility to true
            self.current_arrow_visualizer.set_visibility(True)
            if not hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer = VisualizationMarkers(self.cfg.future_arrow_visualizer_cfg)
            self.future_arrow_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_goal_pose_visualizer"):
                self.current_goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_poses_visualizer"):
                self.future_goal_poses_visualizer.set_visibility(False)

            if hasattr(self, "current_arrow_visualizer"):
                self.current_arrow_visualizer.set_visibility(False)
            if hasattr(self, "future_arrow_visualizer"):
                self.future_arrow_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # 只显示第一个环境（env_id=0）的可视化，避免多环境共享赛道时的视觉混乱
        env_id = 0
        
        # 获取第一个环境的目标点数据
        env_0_goals = self.env_goals[env_id:env_id+1, : self.num_goals]  # shape: (1, num_goals, 3)
        env_0_xy_goals = env_0_goals[:, :, :2] + self.env_origins[env_id:env_id+1, :2].unsqueeze(1)
        env_0_goal_height = self.env_per_heights[env_id:env_id+1].unsqueeze(-1) * self.terrain.cfg.terrain_generator.vertical_scale
        env_0_goal_pos = torch.concat([env_0_xy_goals, env_0_goal_height], dim=-1)  # shape: (1, num_goals, 3)
        
        # 获取第一个环境的目标状态
        env_0_future_goal_idx = self.future_goal_idx[env_id:env_id+1, :]  # shape: (1, num_goals)
        env_0_current_goal_pos = env_0_goal_pos[~env_0_future_goal_idx, :]  # 已到达的目标（绿色）
        env_0_future_goal_pos = env_0_goal_pos[env_0_future_goal_idx, :]  # 未到达的目标（红色）
        
        # 可视化目标点
        if len(env_0_current_goal_pos) > 0:
            self.current_goal_pose_visualizer.visualize(
                translations=env_0_current_goal_pos,
            )
        if len(env_0_future_goal_pos) > 0:
            self.future_goal_poses_visualizer.visualize(
                translations=env_0_future_goal_pos,
            )
        
        # 获取第一个环境的箭头数据
        env_0_target_pos_rel = self.target_pos_rel[env_id:env_id+1, :]  # shape: (1, 2)
        env_0_next_target_pos_rel = self.next_target_pos_rel[env_id:env_id+1, :]  # shape: (1, 2)
        env_0_robot_pos_w = self.robot.data.root_pos_w[env_id:env_id+1, :]  # shape: (1, 3)
        
        current_arrow_list = []
        future_arrow_list = []
        for i in range(self.arrow_num):
            # 当前目标箭头
            norm = torch.norm(env_0_target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = env_0_target_pos_rel / (norm + 1e-5)
            current_pose_arrow = env_0_robot_pos_w[:, :2] + 0.1*(i+3) * target_vec_norm[:, :2]
            current_arrow_list.append(torch.concat([
                current_pose_arrow[:,0][:,None], 
                current_pose_arrow[:,1][:,None], 
                env_0_robot_pos_w[:, 2:3]
                ], dim = 1))
            
            # 下一个目标箭头
            if len(env_0_future_goal_pos) > 0:
                norm = torch.norm(env_0_next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = env_0_next_target_pos_rel / (norm + 1e-5)
                future_pose_arrow = env_0_robot_pos_w[:, :2] + 0.2*(i+3) * target_vec_norm[:, :2]
                future_arrow_list.append(torch.concat([
                    future_pose_arrow[:,0][:,None], 
                    future_pose_arrow[:,1][:,None], 
                    env_0_robot_pos_w[:, 2:3]
                    ], dim = 1))
            else:
                # 如果没有未来目标，使用当前目标箭头
                future_arrow_list.append(torch.concat([
                    current_pose_arrow[:,0][:,None], 
                    current_pose_arrow[:,1][:,None], 
                    env_0_robot_pos_w[:, 2:3]
                    ], dim = 1))

        # 可视化箭头
        if len(current_arrow_list) > 0:
            current_arrow_positions = torch.cat(current_arrow_list, dim=0)
            self.current_arrow_visualizer.visualize(
                translations=current_arrow_positions,
            )
        
        if len(future_arrow_list) > 0:
            future_arrow_positions = torch.cat(future_arrow_list, dim=0)
            self.future_arrow_visualizer.visualize(
                translations=future_arrow_positions,
            )

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")
