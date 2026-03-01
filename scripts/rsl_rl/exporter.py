# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import numpy as np
import torch


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.array(
            [v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for v in value],
            dtype=object,
        )
    return np.array(value)


def _looks_like_pattern(name: str) -> bool:
    name = str(name)
    if ".*" in name:
        return True
    return any(ch in name for ch in ("*", "?", "[", "]", "|", "^", "$"))


def _infer_actor_input_dim(actor: torch.nn.Module) -> int | None:
    """Infer actor input dim by taking the maximum in_features across Linear layers.

    Some actors contain auxiliary encoders whose first Linear has smaller in_features
    (e.g., scan encoder), so returning the first Linear underestimates the true
    observation size. Using the max is safer for ONNX export.
    """
    max_in = None
    if hasattr(actor, "in_features"):
        try:
            max_in = int(getattr(actor, "in_features"))
        except Exception:
            max_in = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            val = int(module.in_features)
            if (max_in is None) or (val > max_in):
                max_in = val
    return max_in


def _call_actor(actor: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    try:
        return actor(obs, hist_encoding=True)
    except TypeError:
        try:
            return actor(obs)
        except TypeError:
            return actor(obs, True)


# Export override profile to align exported artifacts with the provided YAML spec.
# This is intentionally explicit and acts as the source of truth for export output.
EXPORT_PROFILE_GALILEO = {
    "dt": 0.02,
    "joint_names": [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ],
    "default_joint_pos": [
        -0.0500,
        0.0500,
        -0.0500,
        0.0500,
        0.7500,
        0.7500,
        0.7500,
        0.7500,
        -1.5000,
        -1.5000,
        -1.5000,
        -1.5000,
    ],
    "input_actor_obs_names": [
        "base_ang_vel_nad",
        "projected_gravity_nad",
        "joint_pos_rel_nad",
        "joint_vel_rel_nad",
        "actions_gt",
        "base_commands_gt",
    ],
    "input_actor_obs_scales": {
        "base_ang_vel_nad": 0.25,
        "projected_gravity_nad": 1.0,
        "joint_pos_rel_nad": 1.0,
        "joint_vel_rel_nad": 0.05,
        "actions_gt": 0.25,
        "base_commands_gt": [2.0, 2.0, 0.25],
    },
    "input_obs_size_map": {"actor_obs": 45},
    "action_scale": 0.25,
    "clip_actions": 100.0,
    "clip_obs": 100.0,
    "obs_history_length": {"actor_obs": 1},
    "joint_kp": [70.0] * 12,
    "joint_kd": [3.5] * 12,
    "max_torques": [80.0] * 12,
    "velocity_x_forward_scale": 1.0,
    "velocity_x_backward_scale": 0.8,
    "velocity_y_scale": 0.4,
    "velocity_yaw_scale": 1.0,
    "max_velocity": [1.0, 0.4, 1.5],
    "max_acceleration": [1.5, 1.5, 6.0],
    "max_jerk": [5.0, 5.0, 30.0],
    "threshold": {"limit_lower": -0.0, "limit_upper": 0.0, "damping": 5.0},
}


def _apply_export_overrides(policy_cfg_dict: dict, overrides: dict | None) -> dict:
    if not overrides:
        return policy_cfg_dict
    for key, value in overrides.items():
        policy_cfg_dict[key] = copy.deepcopy(value)
    return policy_cfg_dict


def _extract_action_joint_names(env_cfg):
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return None
    joint_pos_cfg = getattr(actions_cfg, "joint_pos", None)
    if joint_pos_cfg is None:
        return None
    joint_names = getattr(joint_pos_cfg, "joint_names", None)
    if not joint_names:
        return None
    if any(_looks_like_pattern(name) for name in joint_names):
        return None
    return list(joint_names)


def _resolve_obs_group(obs_mgr):
    group_names = list(getattr(obs_mgr, "group_obs_dim", {}) or {})
    if not group_names:
        group_names = list(getattr(obs_mgr, "_group_obs_term_names", {}) or {})
    for name in ("actor_obs", "policy"):
        if name in group_names:
            return name
    return group_names[0] if group_names else None


def _extract_obs_terms(env):
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    if group is None:
        return []
    return list(getattr(obs_mgr, "_group_obs_term_names", {}).get(group, []))


def _extract_obs_scales(env):
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    if group is None:
        return {}
    term_names = list(getattr(obs_mgr, "_group_obs_term_names", {}).get(group, []))
    term_cfgs = list(getattr(obs_mgr, "_group_obs_term_cfgs", {}).get(group, []))
    scales = {}
    for name, cfg in zip(term_names, term_cfgs):
        scale = getattr(cfg, "scale", None)
        if scale is None:
            params = getattr(cfg, "params", None)
            if isinstance(params, dict) and "scale" in params:
                scale = params["scale"]
        if scale is None:
            scale = 1.0
        scales[name] = scale
    return scales


def _infer_command_max_velocity(command_cfg):
    explicit_max_vel = getattr(command_cfg, "max_velocity", None)
    if explicit_max_vel is not None:
        return [float(v) for v in explicit_max_vel]
    max_vel = [1.0, 1.0, 1.0]
    ranges = getattr(command_cfg, "ranges", None)
    if ranges is not None:
        if isinstance(ranges, dict):
            range_iter = ranges.values()
        else:
            range_iter = [ranges]
        for rng in range_iter:
            if isinstance(rng, dict):
                lin_x = rng.get("lin_vel_x")
                lin_y = rng.get("lin_vel_y")
                ang_z = rng.get("ang_vel_z")
            else:
                lin_x = getattr(rng, "lin_vel_x", None)
                lin_y = getattr(rng, "lin_vel_y", None)
                ang_z = getattr(rng, "ang_vel_z", None)
            if lin_x is not None:
                max_vel[0] = max(max_vel[0], max(abs(lin_x[0]), abs(lin_x[1])))
            if lin_y is not None:
                max_vel[1] = max(max_vel[1], max(abs(lin_y[0]), abs(lin_y[1])))
            if ang_z is not None:
                max_vel[2] = max(max_vel[2], max(abs(ang_z[0]), abs(ang_z[1])))
    max_lin_x_level = getattr(command_cfg, "max_lin_x_level", None)
    max_ang_z_level = getattr(command_cfg, "max_ang_z_level", None)
    if max_lin_x_level is not None:
        max_vel[0] = max(max_vel[0], abs(max_lin_x_level))
    if max_ang_z_level is not None:
        max_vel[2] = max(max_vel[2], abs(max_ang_z_level))
    return [float(v) for v in max_vel]


def _extract_joint_defaults(env, joint_names):
    joint_name_to_idx = {name: idx for idx, name in enumerate(env.unwrapped.scene.articulations["robot"].joint_names)}
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()
    return [float(f"{default_joint_pos[joint_name_to_idx[name]]:.4f}") for name in joint_names]


def _extract_actuator_vectors(env, joint_names, fallback_kp=90.0, fallback_kd=3.0, fallback_torque=130.0):
    actuators = getattr(env.unwrapped.scene.articulations["robot"], "actuators", {})
    actuator = None
    if isinstance(actuators, dict):
        actuator = actuators.get("base_legs")
        if actuator is None and actuators:
            actuator = next(iter(actuators.values()))
    kp_list = [fallback_kp for _ in joint_names]
    kd_list = [fallback_kd for _ in joint_names]
    torque_list = [fallback_torque for _ in joint_names]
    if actuator is None:
        return kp_list, kd_list, torque_list
    act_joint_names = getattr(actuator, "joint_names", None)
    stiffness = getattr(actuator, "stiffness", None)
    damping = getattr(actuator, "damping", None)
    effort_limit = getattr(actuator, "effort_limit", None)
    if stiffness is not None:
        stiffness = _to_numpy(stiffness).reshape(-1)
    if damping is not None:
        damping = _to_numpy(damping).reshape(-1)
    if effort_limit is not None:
        effort_limit = _to_numpy(effort_limit).reshape(-1)
    if act_joint_names and stiffness is not None and len(act_joint_names) == len(stiffness):
        kp_map = {name: _safe_float(val, fallback_kp) for name, val in zip(act_joint_names, stiffness)}
        kp_list = [kp_map.get(name, fallback_kp) for name in joint_names]
    elif stiffness is not None and len(stiffness) == len(joint_names):
        kp_list = [_safe_float(val, fallback_kp) for val in stiffness]
    if act_joint_names and damping is not None and len(act_joint_names) == len(damping):
        kd_map = {name: _safe_float(val, fallback_kd) for name, val in zip(act_joint_names, damping)}
        kd_list = [kd_map.get(name, fallback_kd) for name in joint_names]
    elif damping is not None and len(damping) == len(joint_names):
        kd_list = [_safe_float(val, fallback_kd) for val in damping]
    if act_joint_names and effort_limit is not None and len(act_joint_names) == len(effort_limit):
        tq_map = {name: _safe_float(val, fallback_torque) for name, val in zip(act_joint_names, effort_limit)}
        torque_list = [tq_map.get(name, fallback_torque) for name in joint_names]
    elif effort_limit is not None and len(effort_limit) == len(joint_names):
        torque_list = [_safe_float(val, fallback_torque) for val in effort_limit]
    return kp_list, kd_list, torque_list


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_policy_as_onnx_dual_input(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    actor_obs_dim: int | None = None,
    verbose: bool = False,
):
    """Export policy into an ONNX file with a single actor_obs input."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterDualInput(
        actor_critic,
        normalizer,
        actor_obs_dim=actor_obs_dim,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return _call_actor(self.actor, x)

    def forward(self, x):
        return _call_actor(self.actor, self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return _call_actor(self.actor, x), h, c

    def forward(self, x):
        return _call_actor(self.actor, self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            actor_in = _infer_actor_input_dim(self.actor)
            if actor_in is None:
                raise RuntimeError("Unable to infer actor input dimension for ONNX export.")
            obs = torch.zeros(1, actor_in)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _OnnxPolicyExporterDualInput(torch.nn.Module):
    """Exporter that exposes a single actor_obs input."""

    def __init__(
        self,
        actor_critic: object,
        normalizer: object | None,
        actor_obs_dim: int | None,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        if getattr(actor_critic, "is_recurrent", False):
            raise RuntimeError("Dual-input exporter does not support recurrent policies.")
        self.actor = copy.deepcopy(actor_critic.actor)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        inferred = _infer_actor_input_dim(self.actor)
        # resolve actor_obs_dim: prefer user-provided; fallback to inferred; ensure >= inferred if inferred is known
        if actor_obs_dim is not None:
            try:
                actor_obs_dim = int(actor_obs_dim)
            except Exception:
                actor_obs_dim = None
        if actor_obs_dim is None or actor_obs_dim <= 0:
            actor_obs_dim = inferred
        if actor_obs_dim is None or actor_obs_dim <= 0:
            raise RuntimeError(
                f"Unable to infer actor_obs_dim (provided={actor_obs_dim}, inferred={inferred}). "
                "Please provide actor_obs_dim explicitly."
            )
        if inferred is not None and actor_obs_dim < inferred:
            actor_obs_dim = inferred

        self.actor_obs_dim = actor_obs_dim
        self.expected_obs_dim = inferred or self.actor_obs_dim

    def _pad_or_trim(self, obs: torch.Tensor) -> torch.Tensor:
        """Ensure obs has expected_obs_dim without tracing-time python branching.

        During ONNX export / tracing, shape values may be Tensors; any python
        branching on them triggers TracerWarning. In tracing, we simply slice to
        the expected dimension (slicing is ONNX-friendly). Outside tracing, keep
        the original pad/trim logic for robustness.
        """
        expected = self.actor_obs_dim or self.expected_obs_dim
        if expected is None:
            return obs
        # Avoid python bool on Tensor when tracing/onnx export
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return obs[:, :expected]

        current = obs.shape[1]
        if current == expected:
            return obs
        if current > expected:
            return obs[:, :expected]
        pad = expected - current
        if pad <= 0:
            return obs
        zeros = torch.zeros(obs.shape[0], pad, device=obs.device, dtype=obs.dtype)
        return torch.cat([obs, zeros], dim=1)

    def forward(self, actor_obs):
        obs = self._pad_or_trim(actor_obs)
        return _call_actor(self.actor, self.normalizer(obs))

    def export(self, path, filename):
        self.to("cpu")
        actor_dim = self.actor_obs_dim or self.expected_obs_dim
        if actor_dim is None:
            raise RuntimeError("Unable to infer actor_obs_dim for ONNX export.")
        actor_obs = torch.zeros(1, actor_dim)
        torch.onnx.export(
            self,
            actor_obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["actor_obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_inference_cfg(
    env, env_cfg, path, load_run, checkpoint, agent_cfg=None, export_overrides: dict | None = None
):
    policy_cfg_dict = {}
    physics_dt = getattr(env.unwrapped, "physics_dt", None)
    if physics_dt is not None:
        policy_cfg_dict["dt"] = env_cfg.decimation * physics_dt
    else:
        policy_cfg_dict["dt"] = (
            getattr(env.unwrapped, "step_dt", None)
            or getattr(getattr(env.unwrapped, "sim", None), "dt", None)
            or 0.0
        )

    joint_names = _extract_action_joint_names(env_cfg)
    if not joint_names:
        joint_names = _extract_action_joint_names(env.unwrapped.cfg)
    if not joint_names:
        joint_names = list(env.unwrapped.scene.articulations["robot"].joint_names)
    policy_cfg_dict["joint_names"] = joint_names
    policy_cfg_dict["default_joint_pos"] = _extract_joint_defaults(env, joint_names)

    policy_cfg_dict["input_names"] = ["actor_obs"]
    policy_cfg_dict["output_names"] = ["actions"]

    actor_obs_names = _extract_obs_terms(env)
    summary = getattr(env.unwrapped.cfg, "config_summary", None) or getattr(env_cfg, "config_summary", None)
    env_summary = getattr(summary, "env", None)
    obs_summary = getattr(summary, "observation", None)
    action_summary = getattr(summary, "action", None)

    action_history_length = getattr(env_summary, "action_history_length", None)
    if action_history_length is None:
        action_history_length = getattr(
            getattr(getattr(env_cfg, "actions", None), "joint_pos", None), "history_length", 1
        )
    include_action_hist = getattr(env_summary, "include_action_hist", None)
    if include_action_hist is None:
        include_action_hist = action_history_length > 1
    if include_action_hist and "action_hist" not in actor_obs_names:
        actor_obs_names.append("action_hist")
    policy_cfg_dict["input_actor_obs_names"] = actor_obs_names

    obs_scales = _extract_obs_scales(env)
    obs_name_map = getattr(obs_summary, "export_name_map", None)
    if isinstance(obs_name_map, dict) and obs_name_map:
        mapped_names = []
        mapped_scales = {}
        for name in actor_obs_names:
            mapped = obs_name_map.get(name, name)
            mapped_names.append(mapped)
            if name in obs_scales:
                mapped_scales[mapped] = obs_scales[name]
        actor_obs_names = mapped_names
        obs_scales = mapped_scales
        policy_cfg_dict["input_actor_obs_names"] = actor_obs_names
    for name in actor_obs_names:
        if name not in obs_scales:
            obs_scales[name] = 1.0

    def _normalize_scale(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return [_safe_float(v, 1.0) for v in val]
        return _safe_float(val, 1.0)

    input_actor_obs_scales = {name: _normalize_scale(obs_scales[name]) for name in actor_obs_names}
    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    obs_mgr = env.unwrapped.observation_manager
    group = _resolve_obs_group(obs_mgr)
    group_dims = getattr(obs_mgr, "group_obs_dim", {})
    actor_obs_dim = None
    if group is not None and group in group_dims:
        dim = group_dims[group]
        actor_obs_dim = int(dim[0] if isinstance(dim, (list, tuple)) else dim)
    if actor_obs_dim is None:
        actor_obs_dim = int(len(actor_obs_names))
    policy_cfg_dict["input_obs_size_map"] = {"actor_obs": int(actor_obs_dim)}
    action_scale = getattr(action_summary, "scale", None)
    if action_scale is None:
        action_scale = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        action_scale = getattr(action_scale, "scale", 1.0)
    policy_cfg_dict["action_scale"] = _safe_float(action_scale, 1.0)

    clip_actions = getattr(env_summary, "clip_actions", None)
    if clip_actions is None and agent_cfg is not None:
        clip_actions = getattr(agent_cfg, "clip_actions", None)
    if clip_actions is None:
        clip_actions = 1.0
    policy_cfg_dict["clip_actions"] = _safe_float(clip_actions, 1.0)

    clip_obs = getattr(env_summary, "clip_obs", None)
    if clip_obs is None:
        clip_obs = 100.0
    policy_cfg_dict["clip_obs"] = _safe_float(clip_obs, 100.0)
    policy_cfg_dict["obs_history_length"] = {"actor_obs": 1}

    kp, kd, max_torques = _extract_actuator_vectors(env, joint_names)
    policy_cfg_dict["joint_kp"] = [float(f"{x:.4f}") for x in kp]
    policy_cfg_dict["joint_kd"] = [float(f"{x:.4f}") for x in kd]
    policy_cfg_dict["max_torques"] = [float(f"{x:.4f}") for x in max_torques]

    command_cfg = getattr(summary, "command", None)
    if command_cfg is None:
        command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if command_cfg is None:
        command_cfg = getattr(env.unwrapped.cfg, "commands", None)
    policy_cfg_dict["velocity_x_forward_scale"] = _safe_float(
        getattr(command_cfg, "velocity_x_forward_scale", 1.0) if command_cfg else 1.0, 1.0
    )
    policy_cfg_dict["velocity_x_backward_scale"] = _safe_float(
        getattr(command_cfg, "velocity_x_backward_scale", 1.0) if command_cfg else 1.0, 1.0
    )
    policy_cfg_dict["velocity_y_scale"] = _safe_float(
        getattr(command_cfg, "velocity_y_scale", 1.0) if command_cfg else 1.0, 1.0
    )
    policy_cfg_dict["velocity_yaw_scale"] = _safe_float(
        getattr(command_cfg, "velocity_yaw_scale", 1.0) if command_cfg else 1.0, 1.0
    )
    policy_cfg_dict["max_velocity"] = (
        _infer_command_max_velocity(command_cfg) if command_cfg is not None else [1.0, 1.0, 1.0]
    )
    policy_cfg_dict["max_acceleration"] = [1.5, 1.5, 6.0]
    policy_cfg_dict["max_jerk"] = [5.0, 5.0, 30.0]
    policy_cfg_dict["threshold"] = {"limit_lower": 0.0, "limit_upper": 0.0, "damping": 5.0}
    if export_overrides is None:
        export_overrides = EXPORT_PROFILE_GALILEO
    policy_cfg_dict = _apply_export_overrides(policy_cfg_dict, export_overrides)

    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    print("max_torques:", policy_cfg_dict["max_torques"])
    print("velocity_x_forward_scale:", policy_cfg_dict["velocity_x_forward_scale"])
    print("velocity_x_backward_scale:", policy_cfg_dict["velocity_x_backward_scale"])
    print("velocity_y_scale:", policy_cfg_dict["velocity_y_scale"])
    print("velocity_yaw_scale:", policy_cfg_dict["velocity_yaw_scale"])
    print("max_velocity:", policy_cfg_dict["max_velocity"])
    print("max_acceleration:", policy_cfg_dict["max_acceleration"])
    print("max_jerk:", policy_cfg_dict["max_jerk"])
    print("threshold:", policy_cfg_dict["threshold"])
    export_inference_cfg_to_yaml(policy_cfg_dict, path, load_run, checkpoint)
    return policy_cfg_dict


def export_inference_cfg_to_yaml(config_dict, path, load_run, checkpoint):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    content += "input_obs_names_map:\n  {\n"
    content += "    actor_obs: ["
    content += ", ".join(f'"{o}"' for o in config_dict["input_actor_obs_names"])
    content += "],\n  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    scales = config_dict["input_actor_obs_scales"]
    obs_list = config_dict["input_actor_obs_names"]
    content += "input_obs_scales_map:\n  {\n    actor_obs: { "
    parts = []
    for obs in obs_list:
        val = scales.get(obs, 1.0)
        if isinstance(val, list):
            sval = "[" + ", ".join(f"{x}" for x in val) + "]"
        else:
            sval = f"{val}"
        parts.append(f"{obs}: {sval}")
    content += ", ".join(parts)
    content += " },\n  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, dim in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {dim},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"
    content += f"velocity_x_forward_scale: {config_dict['velocity_x_forward_scale']}\n"
    content += f"velocity_x_backward_scale: {config_dict['velocity_x_backward_scale']}\n"
    content += f"velocity_y_scale: {config_dict['velocity_y_scale']}\n"
    content += f"velocity_yaw_scale: {config_dict['velocity_yaw_scale']}\n"
    content += "max_velocity: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_velocity"])
    content += "]\n"
    content += "max_acceleration: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_acceleration"])
    content += "]\n"
    content += "max_jerk: ["
    content += ", ".join(f"{float(v)}" for v in config_dict["max_jerk"])
    content += "]\n"
    content += "threshold:\n"
    content += f"  limit_lower: {config_dict['threshold']['limit_lower']}\n"
    content += f"  limit_upper: {config_dict['threshold']['limit_upper']}\n"
    content += f"  damping: {config_dict['threshold']['damping']}\n"
    content += f"max_torques: {config_dict['max_torques']}\n"
    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)
