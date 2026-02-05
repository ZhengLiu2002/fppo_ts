# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations
import torchvision
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from crl_isaaclab.envs.mdp.crl_events import CRLEvent 
from collections.abc import Sequence
import numpy as np 
import cv2
import isaaclab.envs.mdp as mdp
if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ExtremeCRLObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: CRLManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.crl_event: CRLEvent =  env.crl_manager.get_term(cfg.params["crl_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params['history_length']
        self.include_privileged = cfg.params.get("include_privileged", True)

        # 延迟初始化，保证维度与实时观测对齐（兼容新增特征）。
        self._obs_history_buffer = None
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env
        # Galileo uses base_link; keep regex flexible.
        base_candidates = ["base", "base_link"]
        for cand in base_candidates:
            try:
                self.body_id = self.asset.find_bodies(cand)[0]
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Cannot find base body with patterns {base_candidates}")
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._obs_history_buffer is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._obs_history_buffer[env_ids, :, :] = 0. 

    def __call__(
        self,
        env: CRLManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        crl_name: str,
        history_length: int,
        ) -> torch.Tensor:
        
        terrain_names = self.crl_event.env_per_terrain_name
        env_idx_tensor = torch.tensor((terrain_names != 'crl_flat')).to(dtype = torch.bool, device=self.device)
        invert_env_idx_tensor = torch.tensor((terrain_names == 'crl_flat')).to(dtype = torch.bool, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.crl_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.crl_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
        commands = env.command_manager.get_command('base_velocity')
        obs_buf = torch.cat((
                            self.asset.data.root_ang_vel_b * 0.25,   #[1,3] 0~2
                            imu_obs,    #[1,2] 3~4
                            self.delta_yaw[:, None],   #[1,1] 5 - 修复：恢复航向误差观测
                            self.delta_yaw[:, None], #[1,1] 6
                            self.delta_next_yaw[:, None], #[1,1] 7 
                            commands[:, 0:2], #[1,2] 8 - 修复：恢复命令的x和y分量
                            commands[:, 0:1],  #[1,1] 9
                            env_idx_tensor,
                            invert_env_idx_tensor,
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel * 0.05 ,
                            env.action_manager.get_term('joint_pos').action_history_buf[:, -1],
                            self._get_contact_fill(),
                            ),dim=-1)
        extra_terms: list[torch.Tensor] = []
        if self.include_privileged:
            priv_explicit = self._get_priv_explicit()
            priv_latent = self._get_priv_latent()
            priv_hurdles = self._get_priv_hurdles(env)
            extra_terms.extend([priv_hurdles, priv_explicit, priv_latent])
        if self._obs_history_buffer is None or self._obs_history_buffer.shape[2] != obs_buf.shape[1]:
            self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, obs_buf.shape[1], device=self.device)
        observations = torch.cat(
            [obs_buf, self.measured_heights, *extra_terms, self._obs_history_buffer.view(self.num_envs, -1)],
            dim=-1,
        )
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return observations 

    def _get_contact_fill(
        self,
        ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)
    
    def _get_priv_explicit(
        self,
        ):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)

    def _get_priv_latent(
        self,
        ):
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)
        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            (joint_stiffness/ default_joint_stiffness) - 1, 
            (joint_damping/ default_joint_damping) - 1
        ), dim=-1).to(self.device)

    def _get_priv_hurdles(self, env):
        """Privileged hurdle heights and modes, normalized to [-1, 1]."""
        heights = self._get_hurdle_heights(env)  # (N, 4) or -1 when absent
        heights_norm = torch.clamp(heights / 0.6, -1.0, 1.0)
        modes = getattr(env.scene, "hurdle_modes", None)
        if modes is None:
            modes_norm = torch.full_like(heights_norm, -1.0)
        else:
            modes_norm = torch.clamp(modes.to(self.device).float(), -1.0, 1.0)
        return torch.cat([heights_norm, modes_norm], dim=-1)
    
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)

    def _get_hurdle_heights(self, env):
        """Privileged hurdle heights (up to 4 bars), filled with -1 when absent."""
        default = torch.full((self.num_envs, 4), -1.0, device=self.device)
        heights = getattr(env.scene, "hurdle_heights", default)
        return heights


class PolicyHistory(ManagerTermBase):
    """History buffer for policy-proprioceptive features."""
    def __init__(self, cfg: ObservationTermCfg, env: CRLManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.history_length = int(cfg.params.get("history_length", 1))
        self.include_base_lin_vel = bool(cfg.params.get("include_base_lin_vel", True))
        self.command_name = cfg.params.get("command_name", "base_velocity")
        self._obs_history_buffer = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._obs_history_buffer is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._obs_history_buffer[env_ids, :, :] = 0.0

    def __call__(
        self,
        env: CRLManagerBasedRLEnv,
        history_length: int,
        include_base_lin_vel: bool,
        command_name: str,
    ) -> torch.Tensor:
        hist_len = int(history_length)
        prop_terms: list[torch.Tensor] = []
        if include_base_lin_vel:
            prop_terms.append(mdp.base_lin_vel(env))
        prop_terms.extend(
            [
                mdp.base_ang_vel(env),
                mdp.projected_gravity(env),
                mdp.joint_pos_rel(env),
                mdp.joint_vel_rel(env),
                mdp.last_action(env),
                mdp.generated_commands(env, command_name=command_name),
            ]
        )
        prop = torch.cat(prop_terms, dim=-1)

        if (
            self._obs_history_buffer is None
            or self._obs_history_buffer.shape[1] != hist_len
            or self._obs_history_buffer.shape[2] != prop.shape[1]
        ):
            self._obs_history_buffer = torch.zeros(self.num_envs, hist_len, prop.shape[1], device=self.device)

        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None],
            torch.stack([prop] * hist_len, dim=1),
            torch.cat([self._obs_history_buffer[:, 1:], prop.unsqueeze(1)], dim=1),
        )
        return self._obs_history_buffer.view(self.num_envs, -1)

class image_features(ManagerTermBase):
    
    def __init__(self, cfg: ObservationTermCfg, env: CRLManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_sensor: RayCasterCamera = env.scene[cfg.params["sensor_cfg"].name]
        self.clipping_range = self.camera_sensor.cfg.max_distance
        resized = cfg.params["resize"]
        self.buffer_len = cfg.params['buffer_len']
        self.debug_vis = cfg.params['debug_vis']
        self.resize_transform = torchvision.transforms.Resize(
                                    (resized[0], resized[1]), 
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(env.device)
        self.depth_buffer = torch.zeros(self.num_envs,  
                                        self.buffer_len, 
                                        resized[0], 
                                        resized[1]).to(self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)[env_ids]
        for depth_image, env_id in zip(depth_images, env_ids):
            processed_image = self._process_depth_image(depth_image)
            self.depth_buffer[env_id] = torch.stack([processed_image]* 2, dim=0)

    def __call__(
        self,
        env: CRLManagerBasedRLEnv,        
        sensor_cfg: SceneEntityCfg,
        resize: tuple(int,int), 
        buffer_len: int,
        debug_vis:bool
        ):
        if env.common_step_counter % 5 == 0:
            depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)
            for env_id, depth_image in enumerate(depth_images):
                processed_image = self._process_depth_image(depth_image)
                self.depth_buffer[env_id] = torch.cat([self.depth_buffer[env_id, 1:], 
                                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
        if self.debug_vis:
            depth_images_np = self.depth_buffer[:, -2].detach().cpu().numpy()
            depth_images_norm = []
            for img in depth_images_np:
                depth_images_norm.append(img)
            rows = []
            ncols = 4
            for i in range(0, len(depth_images_norm), ncols):
                row = np.hstack(depth_images_norm[i:i+ncols])  
                rows.append(row)

            grid_img = np.vstack(rows)   
            cv2.imshow("depth_images_grid", grid_img)
            cv2.waitKey(1)
        return self.depth_buffer[:, -2].to(env.device)

    def _process_depth_image(self, depth_image):
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self._normalize_depth_image(depth_image)
        return depth_image

    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def _normalize_depth_image(self, depth_image):
        depth_image = depth_image  # make similiar to scandot 
        depth_image = (depth_image) / (self.clipping_range)  - 0.5
        return depth_image
    
class obervation_delta_yaw_ok(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: CRLManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

    def __call__(
        self,
        env: CRLManagerBasedRLEnv,    
        crl_name: str,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env.common_step_counter % 5 == 0:
            crl_event: CRLEvent =  env.crl_manager.get_term(crl_name)
            asset: Articulation = env.scene[asset_cfg.name]
            _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
            self.delta_yaw = crl_event.target_yaw - wrap_to_pi(yaw)
        return self.delta_yaw < threshold


def _find_body_id(asset: Articulation, body_name: str) -> int:
    try:
        return asset.find_bodies(body_name)[0]
    except Exception:
        # fallback to common base body names
        for cand in ("base_link", "base"):
            try:
                return asset.find_bodies(cand)[0]
            except Exception:
                continue
    raise ValueError(f"Cannot find body with name pattern '{body_name}' (or fallback base/base_link).")


def base_mass(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
    normalize: bool = False,
    mass_delta_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Return base body mass as (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _find_body_id(asset, body_name)
    masses = asset.root_physx_view.get_masses()[:, body_id].to(env.device)
    masses = masses.unsqueeze(1)
    if not normalize:
        return masses
    if mass_delta_range is None:
        return masses
    low, high = float(mass_delta_range[0]), float(mass_delta_range[1])
    half_range = max(abs(low), abs(high), 1.0e-6)
    default_masses = getattr(asset.data, "default_mass", None)
    if default_masses is None:
        ref = masses.mean()
    else:
        default_masses_t = torch.as_tensor(default_masses, device=env.device)
        if default_masses_t.dim() == 1:
            ref = default_masses_t[body_id]
        else:
            ref = default_masses_t[0, body_id]
    delta = masses - ref
    return torch.clamp(delta / half_range, -1.0, 1.0)


def base_com(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
    normalize: bool = False,
    com_range: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Return base body CoM (body frame) as (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _find_body_id(asset, body_name)
    com = asset.data.com_pos_b[:, body_id, :].to(env.device)
    if not normalize or com_range is None:
        return com
    ranges = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    mins = torch.tensor([r[0] for r in ranges], device=env.device)
    maxs = torch.tensor([r[1] for r in ranges], device=env.device)
    mid = (mins + maxs) * 0.5
    half = torch.clamp((maxs - mins) * 0.5, min=1.0e-6)
    return torch.clamp((com - mid) / half, -1.0, 1.0)


def ground_friction(
    env: CRLManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize: bool = False,
    friction_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Return ground friction coefficient as (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    friction = asset.root_physx_view.get_material_properties()[:, 0, 0].to(env.device).unsqueeze(1)
    if not normalize or friction_range is None:
        return friction
    low, high = float(friction_range[0]), float(friction_range[1])
    mid = (low + high) * 0.5
    half = max((high - low) * 0.5, 1.0e-6)
    return torch.clamp((friction - mid) / half, -1.0, 1.0)
