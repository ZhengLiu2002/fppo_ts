from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

# --- Galileo 跨栏几何尺寸 (与 galileo_parkour 资源匹配) ---
HURDLE_BAR_LENGTH = 1.6  # 跨栏杆的长度 (米)
HURDLE_BAR_THICKNESS = 0.07  # 跨栏杆的厚度/直径 (米)
HURDLE_BASE_THICKNESS = 0.04  # 跨栏底座的厚度 (米)

# --- 为 Galileo 身体尺寸调整的默认引导参数 ---
GUIDANCE_LANE_HALF_WIDTH = 0.4  # 引导车道的一半宽度 (米)
GUIDANCE_BACK_SENSE = 0.65  # 向后感知/检测的距离 (米)
GUIDANCE_DETECTION_RANGE = 1.2  # 障碍物检测范围 (米)
OBSTACLE_HEIGHT_THRESHOLD = 0.35  # 确定为障碍物的最小高度阈值 (米)
HEIGHT_TARGET_FLAT = 0.4  # 平地模式下的目标高度 (米)
HEIGHT_TARGET_CRAWL_MIN = 0.25  # 爬行模式下的最小目标高度 (米)
HEIGHT_TARGET_CRAWL_MAX = 0.34  # 爬行模式下的最大目标高度 (米)
HEIGHT_TOLERANCE = 0.04  # 高度容差/允许的误差 (米)
GUIDANCE_SPEED_GATE = 0.18  # 引导速度阈值/门限 (米/秒)
JUMP_WINDOW_FRONT = 0.55  # 跳跃窗口的前沿距离 (在障碍物前) (米)
JUMP_WINDOW_BACK = -0.2  # 跳跃窗口的后沿距离 (在障碍物后) (米)
JUMP_SAFETY_MARGIN = 0.08  # 跳跃安全裕度 (米)
FEET_CLEARANCE_MARGIN = 0.05  # 脚部间隙裕度 (米)
FEET_CLEARANCE_X = 0.28  # 脚部在 X 轴上的清除/间隙距离 (米)
FEET_CLEARANCE_Y = HURDLE_BAR_LENGTH * 0.53  # 脚部在 Y 轴上的清除/间隙距离 (基于栏杆长度) (米)
TRAVERSAL_LAT_THRESHOLD = 0.35  # 穿越时的横向阈值 (米)
TRAVERSAL_BACK_WINDOW = 0.55  # 穿越时的后退/向后窗口距离 (米)

MODE_FLAT = 0  # 模式: 平地/正常行走
MODE_JUMP = 1  # 模式: 跳跃跨越
MODE_CRAWL = 2  # 模式: 爬行通过
HURDLE_LAYOUT_NONE = -1
HURDLE_LAYOUT_JUMP = 0
HURDLE_LAYOUT_CRAWL = 1


def reward_alive(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small survival bonus to keep agents exploring instead of instant resets."""
    return torch.ones(env.num_envs, device=env.device)


def _get_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return body yaw from quaternion."""
    _, _, yaw = euler_xyz_from_quat(quat)
    return yaw


def _default_hurdle_spacing(env):
    """Read hurdle spacing/start from the reset event."""
    start, spacing = 2.0, 2.0
    try:
        place_cfg = getattr(env.cfg.events, "place_hurdles", None)
        if place_cfg is not None:
            start = place_cfg.params.get("start", start)
            spacing = place_cfg.params.get("spacing", spacing)
    except Exception:
        pass
    return float(start), float(spacing)


def _get_hurdle_layout(
    env,
    base_thickness: float = HURDLE_BASE_THICKNESS,
    bar_thickness: float = HURDLE_BAR_THICKNESS,
):
    """获取每个 env 栏杆的世界系坐标与高度。

    - 依赖重置事件写入的 hurdle_heights/hurdle_modes 缓存，避免在奖励函数内重复推断。
    - 计算栏杆顶面高度时加上底座和横杆厚度，确保与实际碰撞高度一致。
    """
    heights = getattr(env.scene, "hurdle_heights", None)
    if heights is None:
        return None

    valid_mask = heights >= 0.0
    if not torch.any(valid_mask):
        return None
    modes = getattr(env.scene, "hurdle_modes", None)

    start, spacing = _default_hurdle_spacing(env)
    idxs = torch.arange(
        heights.shape[1], device=env.device, dtype=env.scene.env_origins.dtype
    )
    start_t = torch.tensor(start, device=env.device, dtype=env.scene.env_origins.dtype)
    spacing_t = torch.tensor(
        spacing, device=env.device, dtype=env.scene.env_origins.dtype
    )
    # 栏杆沿 X 轴依次排列，Y 轴对齐 env 原点；地面高度来自 terrain env_origins
    x_positions = env.scene.env_origins[:, 0:1] + start_t + spacing_t * idxs
    x_positions = x_positions.expand_as(heights)
    y_positions = env.scene.env_origins[:, 1:2].expand_as(heights)
    ground_z = env.scene.env_origins[:, 2:3]
    # bar_top_z includes base and bar thickness to reflect actual collision height
    bar_top_z = ground_z + base_thickness + torch.clamp(heights, min=0.0) + bar_thickness
    layout = {
        "x": x_positions,
        "y": y_positions,
        "top_z": bar_top_z,
        "heights": heights,
        "valid_mask": valid_mask,
    }
    if modes is not None:
        layout["modes"] = modes
    return layout


def _get_hurdle_cache(
    env,
    asset: Articulation,
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_extend: float = GUIDANCE_BACK_SENSE,
    base_thickness: float = HURDLE_BASE_THICKNESS,
    bar_thickness: float = HURDLE_BAR_THICKNESS,
):
    """缓存栏杆相对几何（前向/横向/高度等），避免多奖励重复计算。

    - 以机器人 base 为参考，将世界系坐标旋转到机体局部坐标，得到前向/横向距离。
    - 只保留车道宽度内、且在机器人后向 back_extend 范围前的栏杆。
    - 对每个 env 选取前向距离绝对值最小的栏杆，作为“当前最近栏杆”供后续使用。
    """
    step = getattr(env, "common_step_counter", None)
    cache = getattr(env, "_hurdle_cache", None)
    if (
        cache is not None
        and cache.get("step") == step
        and cache.get("lane_half_width") == lane_half_width
        and cache.get("back_extend") == back_extend
    ):
        return cache

    layout = _get_hurdle_layout(env, base_thickness, bar_thickness)
    if layout is None:
        env._hurdle_cache = {
            "step": step,
            "has_any": False,
            "valid_mask": None,
            "layout": None,
        }
        return env._hurdle_cache

    root_xy = asset.data.root_pos_w[:, :2]
    yaw = _get_yaw(asset.data.root_quat_w)
    cos_yaw = torch.cos(yaw).unsqueeze(1)
    sin_yaw = torch.sin(yaw).unsqueeze(1)

    rel_x = layout["x"] - root_xy[:, 0:1]
    rel_y = layout["y"] - root_xy[:, 1:2]
    forward = rel_x * cos_yaw + rel_y * sin_yaw
    lateral = -rel_x * sin_yaw + rel_y * cos_yaw

    valid_mask = layout["valid_mask"] & (torch.abs(lateral) <= lane_half_width) & (
        forward > -back_extend
    )
    has_any = torch.any(valid_mask, dim=1)

    default_far = torch.full_like(forward, 1.0e3)
    dist_abs = torch.where(valid_mask, torch.abs(forward), default_far)
    min_abs, idx = torch.min(dist_abs, dim=1)
    idx_expand = idx.unsqueeze(-1)
    nearest_forward = forward.gather(1, idx_expand).squeeze(-1)
    nearest_lateral = lateral.gather(1, idx_expand).squeeze(-1)
    nearest_top_z = layout["top_z"].gather(1, idx_expand).squeeze(-1)
    nearest_raw_h = layout["heights"].gather(1, idx_expand).squeeze(-1)

    cache = {
        "step": step,
        "layout": layout,
        "forward_distance": forward,
        "lateral_distance": lateral,
        "valid_mask": valid_mask,
        "has_any": has_any,
        "min_abs_dist": min_abs,
        "nearest_forward": nearest_forward,
        "nearest_lateral": nearest_lateral,
        "nearest_top_z": nearest_top_z,
        "nearest_raw_height": nearest_raw_h,
        "yaw": yaw,
        "lane_half_width": lane_half_width,
        "back_extend": back_extend,
    }
    modes = layout.get("modes") if isinstance(layout, dict) else None
    if modes is not None:
        nearest_mode = modes.gather(1, idx_expand).squeeze(-1)
        cache["nearest_mode"] = nearest_mode
    env._hurdle_cache = cache
    return cache


def _get_locomotion_mode(
    env,
    cache: dict | None,
    height_threshold: float = OBSTACLE_HEIGHT_THRESHOLD,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
):
    """基于最近栏杆的高度/距离判定模式：平地、跳跃或钻爬。

    优先使用重置事件标注的 hurdle_modes（jump/crawl），若无标注则按高度阈值自动判断。
    """
    mode = torch.full(
        (env.num_envs,), MODE_FLAT, device=env.device, dtype=torch.long
    )
    if cache is None:
        return mode

    has_any = cache.get("has_any", None)
    if has_any is None:
        return mode
    if isinstance(has_any, bool):
        if not has_any:
            return mode
        has_any_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        has_any_mask = has_any

    if not torch.any(has_any_mask):
        return mode

    is_near = has_any_mask & (cache["min_abs_dist"] < detection_range)
    override = cache.get("nearest_mode", None)
    handled_override = torch.zeros_like(is_near, dtype=torch.bool)
    if override is not None:
        # 赛事/课程可以直接注入模式，避免高度阈值误判
        valid_override = override != HURDLE_LAYOUT_NONE
        crawl_override = valid_override & (override == HURDLE_LAYOUT_CRAWL)
        jump_override = valid_override & (override == HURDLE_LAYOUT_JUMP)
        mode[is_near & crawl_override] = MODE_CRAWL
        mode[is_near & jump_override] = MODE_JUMP
        handled_override = valid_override
    remaining = is_near & (~handled_override)
    if torch.any(remaining):
        is_high = cache["nearest_raw_height"] >= height_threshold
        mode[remaining & is_high] = MODE_CRAWL
        mode[remaining & (~is_high)] = MODE_JUMP
    return mode


class reward_feet_edge(ManagerTermBase):
    """惩罚脚落在地形边缘的接触，防止在跨栏台阶边缘踩空。"""
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        # Support base or base_link naming.
        for cand in ["base", "base_link"]:
            try:
                self.body_id = self.contact_sensor.find_bodies(cand)[0]
                break
            except Exception:
                continue
        else:
            raise ValueError("Cannot find base/body for feet_edge reward; tried base/base_link")
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        ) -> torch.Tensor:
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    height_threshold: float = OBSTACLE_HEIGHT_THRESHOLD,
    flat_scale: float = 1.0,
    crawl_scale: float = 0.1,
    jump_scale: float = 0.01,
    ) -> torch.Tensor: 
    """惩罚关节力矩的绝对值，根据运动模式调节惩罚力度（平地 > 钻爬 > 跳跃）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    base_pen = torch.sum(torch.square(asset.data.applied_torque), dim=1)

    # 根据当前模式降低越障阶段的力矩惩罚，避免抑制必要的大力矩动作
    cache = _get_hurdle_cache(env, asset)
    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=height_threshold,
        detection_range=detection_range,
    )
    scale = torch.ones_like(base_pen) * flat_scale
    scale = torch.where(mode == MODE_CRAWL, torch.as_tensor(crawl_scale, device=env.device), scale)
    scale = torch.where(mode == MODE_JUMP, torch.as_tensor(jump_scale, device=env.device), scale)
    return base_pen * scale

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    """惩罚关节位置偏离默认站立位姿的误差。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    """专门约束髋关节位置，避免过大摆动影响步态对称性。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    """惩罚机体在水平面的角速度，抑制大幅摇摆/滚动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)

class reward_action_rate(ManagerTermBase):
    """惩罚连续两步的动作差异，避免高频抖动。"""
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_actions = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.previous_actions[idx, 0,:] = 0.
        self.previous_actions[idx, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        # 维护长度为 2 的窗口，计算当前动作与上一步的 L2 差
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    """惩罚关节加速度，促使动作变化更平滑。"""
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.previous_joint_vel[idx, 0,:] = 0.
        self.previous_joint_vel[idx, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    detection_range: float = GUIDANCE_DETECTION_RANGE,
) -> torch.Tensor: 
    """惩罚竖直速度（世界系），跳跃/钻爬模式下不施加惩罚，避免与越障奖励冲突。"""
    _ = parkour_name  # 保留参数以兼容配置接口
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(env, asset)
    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=OBSTACLE_HEIGHT_THRESHOLD,
        detection_range=detection_range,
    )
    flat_mask = mode == MODE_FLAT
    if not torch.any(flat_mask):
        return torch.zeros(env.num_envs, device=env.device)
    # Use world-frame vertical speed; body-frame z would penalize tilting as if it were lift.
    rew = torch.square(asset.data.root_vel_w[:, 2])
    return torch.where(flat_mask, rew, torch.zeros_like(rew))

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    """惩罚姿态偏离水平（pitch/roll），仅在平地时启用，以免干扰越障动作。"""
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # Prefer a torch-only gate (avoids numpy->torch device issues).
    try:
        sub_terrains = getattr(env.scene.terrain.cfg.terrain_generator, "sub_terrains", None)
        if sub_terrains is not None and hasattr(parkour_event, "env_class"):
            names = list(sub_terrains.keys())
            if "parkour_flat" in names:
                flat_idx = names.index("parkour_flat")
                is_flat = parkour_event.env_class.to(dtype=torch.long) == flat_idx
                return torch.where(is_flat, rew, torch.zeros_like(rew))
    except Exception:
        pass
    # Fallback: use terrain name strings (CPU) if available.
    terrain_names = getattr(parkour_event, "env_per_terrain_name", None)
    if terrain_names is not None:
        try:
            not_flat = torch.as_tensor(
                (terrain_names != "parkour_flat")[:, -1],
                device=env.device,
                dtype=torch.bool,
            )
            rew = torch.where(not_flat, torch.zeros_like(rew), rew)
        except Exception:
            pass
    return rew


def reward_lane_deviation(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = 1.0,
) -> torch.Tensor:
    """Penalize drifting outside the lane (y-axis offset from env origin)."""
    asset: Articulation = env.scene[asset_cfg.name]
    lateral = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    return -torch.clamp(lateral - lane_half_width, min=0.0)

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
    return rew.float()

def reward_ground_impact(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    """地面冲击惩罚：平滑足端接触力，避免猛烈跺脚（-||f_t - f_{t-1}||^2）。

    不同运动模式给予不同权重（平地 > 钻爬 > 跳跃），在越障时适当放宽惩罚。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_now = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    forces_prev = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    delta = forces_now - forces_prev
    base_penalty = -torch.sum(torch.square(delta), dim=(1, 2))

    # 根据当前越障模式调整惩罚强度：平地 > 钻爬 > 跳跃
    asset: Articulation = env.scene["robot"]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=GUIDANCE_LANE_HALF_WIDTH,
        back_extend=GUIDANCE_BACK_SENSE,
    )
    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=OBSTACLE_HEIGHT_THRESHOLD,
        detection_range=GUIDANCE_DETECTION_RANGE,
    )
    scales = torch.ones(env.num_envs, device=env.device)
    scales = torch.where(mode == MODE_CRAWL, torch.tensor(0.1, device=env.device), scales)
    scales = torch.where(mode == MODE_JUMP, torch.tensor(0.0, device=env.device), scales)

    return base_penalty * scales

class reward_feet_air_time(ManagerTermBase):
    """奖励足端在空中的时间（swing phase），鼓励更自然的抬脚/步态。

    采用 ContactSensor 的 air-time 统计：当某个足端在本时间步内首次接触地面时，
    根据其上一次离地到落地的空中时间进行奖励。
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        min_air_time: float = 0.15,
        max_air_time: float | None = None,
        command_name: str | None = None,
        command_threshold: float = 0.0,
    ) -> torch.Tensor:
        # Fallback if sensor is not configured with track_air_time=True.
        if self.contact_sensor.data.last_air_time is None:
            return torch.zeros(env.num_envs, device=env.device)

        # First contact events for the selected bodies.
        first_contact = self.contact_sensor.compute_first_contact(self.dt)[:, sensor_cfg.body_ids]
        air_time = self.contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

        # Reward only airtime beyond a minimum threshold.
        swing = torch.clamp(air_time - float(min_air_time), min=0.0)
        if max_air_time is not None and max_air_time > min_air_time:
            swing = torch.clamp(swing, max=float(max_air_time - min_air_time))

        # Return a "rate" so that RewardManager's external *dt integration yields per-event reward.
        rew = torch.sum(swing * first_contact, dim=1) / self.dt

        # Optional gate: only reward when command speed is non-trivial.
        if command_name is not None and command_threshold > 0.0:
            try:
                cmd = env.command_manager.get_command(command_name)
                cmd_speed = torch.norm(cmd[:, :2], dim=1)
                rew = rew * (cmd_speed > command_threshold).to(dtype=rew.dtype)
            except Exception:
                pass

        return rew

def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """奖励机体速度在目标方向上的投影，与命令速度对齐后奖励达到上限。"""
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command('base_velocity')[:, 0]
    # 反向运动时给出线性惩罚，避免“倒着走”也能获得其它奖励
    backward_pen = torch.clamp(-proj_vel / (command_vel + 1e-5), 0.0, 1.0)
    forward_term = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
    rew_move = torch.where(proj_vel >= 0.0, forward_term, -backward_pen)
    return rew_move


def reward_standstill_penalty(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    speed_threshold: float = 0.15,
) -> torch.Tensor:
    """惩罚“有速度指令但几乎不动”的站桩行为，防止刷存活/姿态等奖励。"""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        command = env.command_manager.get_command(command_name)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    command_speed = torch.norm(command[:, :2], dim=1)
    planar_speed = torch.norm(asset.data.root_vel_w[:, :2], dim=1)
    penalty = (command_speed > command_threshold) & (planar_speed < speed_threshold)
    return penalty.to(dtype=torch.float, device=env.device)


class reward_goal_progress(ManagerTermBase):
    """鼓励向当前目标点前进，远离目标则产生负奖励。"""
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.prev_dist = torch.zeros(env.num_envs, device=self.device)
        self.parkour_name = cfg.params["parkour_name"]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        parkour_event: ParkourEvent = self.env.parkour_manager.get_term(self.parkour_name)
        dist = torch.norm(parkour_event.target_pos_rel, dim=-1)
        self.prev_dist[idx] = dist[idx]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        parkour_name: str, 
    ) -> torch.Tensor:
        parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
        dist = torch.norm(parkour_event.target_pos_rel, dim=-1)
        progress = self.prev_dist - dist
        self.prev_dist = dist
        return progress


def reward_height_guidance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    height_threshold: float = OBSTACLE_HEIGHT_THRESHOLD,
    target_height: float = HEIGHT_TARGET_FLAT,
    height_tolerance: float = HEIGHT_TOLERANCE,
    speed_gate: float = GUIDANCE_SPEED_GATE,
) -> torch.Tensor:
    """平地高度引导：保持正常站立高度，速度门控避免刷分。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=height_threshold,
        detection_range=detection_range,
    )
    active = mode == MODE_FLAT
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    robot_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # 误差按高斯衰减，偏差越小奖励越大
    error = torch.abs(robot_height - target_height)
    reward = torch.exp(-torch.square(error / (height_tolerance + 1e-6)))

    # 低速时抑制奖励，鼓励在前进中保持目标高度而非“原地刷分”
    planar_speed = torch.norm(asset.data.root_vel_w[:, :2], dim=1)
    vel_gate = torch.clamp(planar_speed / (speed_gate + 1e-6), 0.0, 1.0)
    return reward * active.float() * vel_gate


def reward_crawl_clearance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    height_threshold: float = OBSTACLE_HEIGHT_THRESHOLD,
    target_min_height: float = HEIGHT_TARGET_CRAWL_MIN,
    target_max_height: float = HEIGHT_TARGET_CRAWL_MAX,
    height_tolerance: float = HEIGHT_TOLERANCE,
    clearance_margin: float = 0.06,
    speed_gate: float = 0.25,
) -> torch.Tensor:
    """钻爬模式：越靠近高杆越要求压低机体，并奖励头部/背部留有余量。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=height_threshold,
        detection_range=detection_range,
    )
    active = mode == MODE_CRAWL
    if cache is None or not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    robot_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_raw = cache["nearest_raw_height"]
    # 高杆更苛刻：margin 随高度增大，鼓励 40/50cm 栏杆拉伸身体
    extra_margin = torch.clamp(height_raw - 0.40, min=0.0)
    adaptive_margin = torch.clamp(0.18 + 0.25 * extra_margin, 0.15, 0.30)
    target = torch.clamp(height_raw - adaptive_margin, min=target_min_height, max=target_max_height + 0.05)
    # 越靠近障碍，高度容差越小，引导提前压低
    dist_weight = torch.clamp(1.0 - cache["min_abs_dist"] / detection_range, 0.0, 1.0)
    current_tol = height_tolerance * (1.0 + 2.0 * (1.0 - dist_weight))
    posture_reward = torch.exp(
        -torch.square((robot_height - target) / (current_tol + 1e-6))
    )

    body_max_z = asset.data.body_state_w[:, :, 2].max(dim=1).values
    clearance = cache["nearest_top_z"] - body_max_z + clearance_margin
    clearance_reward = torch.sigmoid(clearance * 30.0)

    height_scale = torch.clamp(height_raw / 0.4, 1.0, 1.6)
    combo = (0.6 * posture_reward + 0.4 * clearance_reward) * height_scale
    yaw = cache.get("yaw", torch.zeros(env.num_envs, device=env.device))
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    forward_speed = asset.data.root_vel_w[:, 0] * cos_yaw + asset.data.root_vel_w[:, 1] * sin_yaw
    forward_speed = torch.clamp(forward_speed, min=0.0)
    vel_gate = torch.clamp(forward_speed / (speed_gate + 1e-6), 0.0, 1.0)
    return active.float() * dist_weight * combo * vel_gate


def reward_jump_clearance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    height_threshold: float = OBSTACLE_HEIGHT_THRESHOLD,
    jump_window_front: float = JUMP_WINDOW_FRONT,
    jump_window_back: float = JUMP_WINDOW_BACK,
    safety_margin: float = JUMP_SAFETY_MARGIN,
    speed_gate: float = 0.25,
) -> torch.Tensor:
    """跳跃模式：在障碍窗口内奖励抬脚高度，按杆高自适应尺度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    if cache is None or cache.get("layout") is None:
        return torch.zeros(env.num_envs, device=env.device)

    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=height_threshold,
        detection_range=detection_range,
    )
    active = mode == MODE_JUMP
    has_hurdle = cache.get("has_any", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    in_window = (
        active
        & has_hurdle
        & (cache["nearest_forward"] <= jump_window_front)
        & (cache["nearest_forward"] >= jump_window_back)
    )
    if not torch.any(in_window):
        return torch.zeros(env.num_envs, device=env.device)

    foot_ids = asset.find_bodies(".*_foot")[0]
    feet_pos = asset.data.body_state_w[:, foot_ids, 2]
    max_foot_height = feet_pos.max(dim=1).values

    height_raw = cache["nearest_raw_height"]
    scaled_margin = safety_margin + 0.30 * torch.clamp(height_raw - 0.20, min=0.0)
    desired_clearance = cache["nearest_top_z"] + scaled_margin
    height_scale = torch.clamp(height_raw / 0.35, 0.8, 1.6)
    clearance = max_foot_height - desired_clearance
    # 高杆期望更高抬脚：height_scale 随杆高线性放大
    reward = torch.sigmoid(clearance * 25.0) * height_scale
    yaw = cache.get("yaw", torch.zeros(env.num_envs, device=env.device))
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    forward_speed = asset.data.root_vel_w[:, 0] * cos_yaw + asset.data.root_vel_w[:, 1] * sin_yaw
    forward_speed = torch.clamp(forward_speed, min=0.0)
    vel_gate = torch.clamp(forward_speed / (speed_gate + 1e-6), 0.0, 1.0)
    return reward * in_window.float() * vel_gate


def reward_feet_clearance(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    check_margin_x: float = FEET_CLEARANCE_X,
    check_margin_y: float = FEET_CLEARANCE_Y,
    safety_margin: float = FEET_CLEARANCE_MARGIN,
) -> torch.Tensor:
    """惩罚在杆附近脚抬得过低，专门防止后腿挂杆。"""
    layout = _get_hurdle_layout(env)
    if layout is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = asset.find_bodies(".*_foot")[0]
    feet_pos = asset.data.body_state_w[:, foot_ids, :3]

    # Broadcast hurdles across feet: feet (N, F, 1), hurdles (N, 1, H)
    # 将脚位置与所有栏杆的 x/y/z 对齐，方便一次性计算所有近邻杆的违例
    layout_x = layout["x"].unsqueeze(1)
    layout_y = layout["y"].unsqueeze(1)
    layout_top_z = layout["top_z"].unsqueeze(1)
    layout_mask = layout["valid_mask"].unsqueeze(1)

    dx = feet_pos[:, :, 0].unsqueeze(-1) - layout_x
    dy = feet_pos[:, :, 1].unsqueeze(-1) - layout_y
    dz = feet_pos[:, :, 2].unsqueeze(-1) - layout_top_z

    near_x = torch.abs(dx) < check_margin_x
    near_y = torch.abs(dy) < check_margin_y
    mask = layout_mask & near_x & near_y
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)

    # 距离越近权重越高；高度不足（dz < safety_margin）按平方惩罚
    closeness = torch.clamp(1.0 - torch.abs(dx) / (check_margin_x + 1e-6), 0.0, 1.0)
    violation_depth = torch.clamp(safety_margin - dz, min=0.0)
    penalty = torch.sum(
        torch.square(violation_depth) * mask.float() * closeness, dim=(1, 2)
    )
    return penalty


def reward_successful_traversal(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    traversal_window: float = TRAVERSAL_BACK_WINDOW,
    lateral_threshold: float = TRAVERSAL_LAT_THRESHOLD,
) -> torch.Tensor:
    """刚越过障碍时对齐奖励，鼓励正向通过而非绕开。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    if cache is None or cache.get("layout") is None:
        return torch.zeros(env.num_envs, device=env.device)

    just_passed = (
        cache["valid_mask"]
        & (cache["forward_distance"] < 0.0)
        & (cache["forward_distance"] > -traversal_window)
    )
    if not torch.any(just_passed):
        return torch.zeros(env.num_envs, device=env.device)

    best_forward = torch.where(
        just_passed,
        cache["forward_distance"],
        torch.full_like(cache["forward_distance"], -1.0e3),
    )
    # 取刚通过的栏杆中前向距离最大的那一个，代表最近通过的杆
    idx = torch.argmax(best_forward, dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    lateral_offset = cache["lateral_distance"][env_ids, idx]
    passed_mask = torch.any(just_passed, dim=1)
    lateral_term = torch.clamp(
        1.0 - torch.abs(lateral_offset) / (lateral_threshold + 1e-6), 0.0, 1.0
    )
    forward_speed = torch.clamp(asset.data.root_vel_w[:, 0], min=0.0)
    speed_term = torch.clamp(forward_speed / 0.4, 0.0, 1.0)
    layout_heights = cache["layout"]["heights"]
    passed_height = layout_heights[env_ids, idx]
    height_scale = torch.clamp(passed_height / 0.35, 0.8, 1.5)
    reward = lateral_term * speed_term * height_scale
    return torch.where(passed_mask, reward, torch.zeros_like(reward))


def reward_mode_mismatch(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    jump_margin: float = JUMP_SAFETY_MARGIN,
    crawl_margin: float = 0.05,
) -> torch.Tensor:
    """轻惩罚错误模式：jump 模式抬脚不足，crawl 模式身体过高。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    if cache is None or cache.get("layout") is None:
        return torch.zeros(env.num_envs, device=env.device)

    mode = _get_locomotion_mode(
        env,
        cache,
        height_threshold=OBSTACLE_HEIGHT_THRESHOLD,
        detection_range=detection_range,
    )
    # jump: 惩罚脚离杆顶过低
    foot_ids = asset.find_bodies(".*_foot")[0]
    feet_pos = asset.data.body_state_w[:, foot_ids, 2]
    max_foot_height = feet_pos.max(dim=1).values
    dyn_jump_margin = torch.clamp(jump_margin + 0.25 * torch.clamp(cache["nearest_raw_height"] - 0.20, min=0.0), 0.08, 0.28)
    desired_jump = cache["nearest_top_z"] + dyn_jump_margin
    jump_deficit = torch.clamp(desired_jump - max_foot_height, min=0.0)
    # Map deficit>=0 to [0, 1) with 0 at deficit=0 (avoid constant 0.5 offset).
    jump_pen = 2.0 * (torch.sigmoid(jump_deficit * 15.0) - 0.5)

    # crawl: 惩罚身体最高点过高
    body_max_z = asset.data.body_state_w[:, :, 2].max(dim=1).values
    dyn_crawl_margin = torch.clamp(crawl_margin + 0.25 * torch.clamp(cache["nearest_raw_height"] - 0.35, min=0.0), 0.05, 0.20)
    crawl_target = cache["nearest_top_z"] - dyn_crawl_margin
    crawl_over = torch.clamp(body_max_z - crawl_target, min=0.0)
    # Same: 0 penalty when under the bar.
    crawl_pen = 2.0 * (torch.sigmoid(crawl_over * 20.0) - 0.5)

    jump_mask = mode == MODE_JUMP
    crawl_mask = mode == MODE_CRAWL
    pen = torch.zeros(env.num_envs, device=env.device)
    pen = torch.where(jump_mask, jump_pen, pen)
    pen = torch.where(crawl_mask, crawl_pen, pen)
    return pen


def reward_low_crawl_penalty(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    low_threshold: float = 0.25,
    posture_margin: float = 0.10,
) -> torch.Tensor:
    """惩罚在低杆（应跳）场景下采取“低身通过”的姿态（无论模式判定如何）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    if cache is None or cache.get("layout") is None:
        return torch.zeros(env.num_envs, device=env.device)

    height_raw = cache["nearest_raw_height"]
    low_mask = height_raw <= low_threshold
    near_mask = cache["has_any"] & low_mask & (cache["min_abs_dist"] < detection_range)
    if not torch.any(near_mask):
        return torch.zeros(env.num_envs, device=env.device)
    # 若试图低身通过低杆：身体最高点低于杆顶+margin → 施加惩罚
    body_max_z = asset.data.body_state_w[:, :, 2].max(dim=1).values
    desired = cache["nearest_top_z"] + posture_margin
    crouch_depth = torch.clamp(desired - body_max_z, min=0.0)
    severity = torch.clamp((low_threshold - height_raw) / (low_threshold + 1e-6), min=0.0, max=1.0)
    # 0 penalty when not crouching under a low hurdle; scale up with how "low" the hurdle is.
    pen = 2.0 * (torch.sigmoid(crouch_depth * 25.0) - 0.5) * (1.0 + severity)
    return torch.where(near_mask, pen, torch.zeros_like(pen))


def reward_jump_success_bonus(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    traversal_window: float = TRAVERSAL_BACK_WINDOW,
) -> torch.Tensor:
    """在跳跃模式成功越过栏杆时给一次性 bonus。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    if cache is None or cache.get("layout") is None:
        return torch.zeros(env.num_envs, device=env.device)
    mode = _get_locomotion_mode(env, cache, height_threshold=OBSTACLE_HEIGHT_THRESHOLD, detection_range=GUIDANCE_DETECTION_RANGE)
    just_passed = (
        cache["valid_mask"]
        & (cache["forward_distance"] < 0.0)
        & (cache["forward_distance"] > -traversal_window)
    )
    if not torch.any(just_passed):
        return torch.zeros(env.num_envs, device=env.device)
    passed_mask = torch.any(just_passed, dim=1) & (mode == MODE_JUMP)
    if not torch.any(passed_mask):
        return torch.zeros(env.num_envs, device=env.device)

    # Identify which hurdle was just passed to evaluate clearance against that bar.
    best_forward = torch.where(
        just_passed,
        cache["forward_distance"],
        torch.full_like(cache["forward_distance"], -1.0e3),
    )
    idx = torch.argmax(best_forward, dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    passed_top_z = cache["layout"]["top_z"][env_ids, idx]

    # bonus scaled by foot clearance after passing
    foot_ids = asset.find_bodies(".*_foot")[0]
    feet_pos = asset.data.body_state_w[:, foot_ids, 2]
    max_foot_height = feet_pos.max(dim=1).values
    clearance = max_foot_height - passed_top_z
    # 越过后抬脚越高，bonus 越大，鼓励完整跳跃而非擦杆
    bonus = torch.sigmoid(clearance * 20.0)
    return torch.where(passed_mask, bonus, torch.zeros_like(bonus))


def reward_lateral_deviation_penalty(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lane_half_width: float = GUIDANCE_LANE_HALF_WIDTH,
    back_sense: float = GUIDANCE_BACK_SENSE,
    detection_range: float = GUIDANCE_DETECTION_RANGE,
    lateral_threshold: float = 0.3,  # 横向偏移阈值，超过此值开始惩罚
    penalty_scale: float = 2.0,  # 惩罚强度
) -> torch.Tensor:
    """惩罚机器人从栏杆边缘绕过：当机器人接近栏杆时，如果横向偏移过大，给予惩罚。
    
    这个奖励函数旨在防止机器人"作弊"从栏杆边缘绕过，鼓励机器人从赛道中心通过栏杆。
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        lane_half_width: 车道半宽
        back_sense: 向后感知距离
        detection_range: 检测范围
        lateral_threshold: 横向偏移阈值，超过此值开始惩罚（米）
        penalty_scale: 惩罚强度系数
    
    Returns:
        惩罚强度（非负，建议配合负权重使用），偏移越大惩罚越大，上限约为 penalty_scale
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = _get_hurdle_cache(
        env,
        asset,
        lane_half_width=lane_half_width,
        back_extend=back_sense,
    )
    
    if cache is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 检查是否有有效的栏杆
    has_any = cache.get("has_any", False)
    if isinstance(has_any, torch.Tensor):
        if not torch.any(has_any):
            return torch.zeros(env.num_envs, device=env.device)
    elif not has_any:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取最近的栏杆信息
    nearest_forward = cache.get("nearest_forward")
    nearest_lateral = cache.get("nearest_lateral")
    
    if nearest_forward is None or nearest_lateral is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 只对在栏杆前方一定范围内的机器人进行惩罚（接近栏杆时）
    # 范围：从栏杆前 detection_range 到栏杆后 back_sense
    near_hurdle_mask = (nearest_forward > -back_sense) & (nearest_forward < detection_range)
    
    if not torch.any(near_hurdle_mask):
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算横向偏移的绝对值
    lateral_abs = torch.abs(nearest_lateral)
    
    # 超过阈值后按车道剩余宽度归一化并截断，避免偶发几何异常导致爆炸值
    # clamp is on a scalar, avoid tensor overload errors
    lane_margin = max(lane_half_width - lateral_threshold, 1e-3)
    excess_lateral = torch.clamp(lateral_abs - lateral_threshold, min=0.0)
    normalized = torch.clamp(excess_lateral / lane_margin, 0.0, 1.0)
    penalty = penalty_scale * torch.square(normalized)
    
    # 只对接近栏杆的机器人应用惩罚
    return torch.where(near_hurdle_mask, penalty, torch.zeros_like(penalty))


def reward_foot_symmetry(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_scale: float = 0.12,
) -> torch.Tensor:
    """鼓励左右成对脚的高度一致，抑制单腿高抬/蹦跳步态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids = asset.find_bodies(".*_foot")[0]
    if len(foot_ids) < 2:
        return torch.ones(env.num_envs, device=env.device)

    foot_pos = asset.data.body_state_w[:, foot_ids, 2]
    # 若少于4只脚，只对现有前两只计算；否则按顺序前(0,1)、后(2,3)配对
    if foot_pos.shape[1] >= 4:
        pairs = [(0, 1), (2, 3)]
    else:
        pairs = [(0, 1)]

    diffs = []
    for a, b in pairs:
        a = min(a, foot_pos.shape[1] - 1)
        b = min(b, foot_pos.shape[1] - 1)
        diffs.append(torch.abs(foot_pos[:, a] - foot_pos[:, b]))

    total_diff = torch.stack(diffs, dim=1).sum(dim=1)
    reward = torch.exp(-total_diff / (height_scale + 1e-6))
    return reward

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """奖励机体航向与目标航向的接近程度，偏差越小奖励越接近 1（使用周期误差）。"""
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # root_quat_w is xyzw; reuse the math helper to avoid component-order mistakes.
    yaw = _get_yaw(asset.data.root_quat_w)
    yaw_error = wrap_to_pi(parkour_event.target_yaw - yaw)
    return torch.exp(-torch.abs(yaw_error))

class reward_delta_torques(ManagerTermBase):
    """惩罚连续步的力矩跳变，防止突然的激烈动作导致不稳定。"""
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.previous_torque[idx, 0,:] = 0.
        self.previous_torque[idx, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # For robots with base_link naming, allow fallback if regex "base" fails.
    if sensor_cfg.body_ids is None or len(sensor_cfg.body_ids) == 0:
        for cand in ["base_link", "base"]:
            try:
                sensor_cfg.body_ids, _ = contact_sensor.find_bodies(cand)
                break
            except Exception:
                continue
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)
