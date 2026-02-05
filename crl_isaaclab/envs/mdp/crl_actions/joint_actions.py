
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import DelayedJointPositionActionCfg

class DelayedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: DelayedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._action_history_buf = torch.zeros(
            self.num_envs, cfg.history_length, self._num_joints, device=self.device, dtype=torch.float
        )
        self._delay_update_global_steps = cfg.delay_update_global_steps
        # normalize delay steps to list
        if isinstance(cfg.action_delay_steps, (list, tuple)):
            self._action_delay_steps = list(cfg.action_delay_steps)
        else:
            self._action_delay_steps = [int(cfg.action_delay_steps)]
        self._use_delay = cfg.use_delay
        self.env = env 
        self.delay = torch.tensor(0, device=self.device, dtype=torch.long)

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        if self._delay_update_global_steps is not None and self._delay_update_global_steps > 0:
            if self.env.common_step_counter % self._delay_update_global_steps == 0:
                if len(self._action_delay_steps) != 0:
                    self.delay = torch.tensor(self._action_delay_steps.pop(0), device=self.device, dtype=torch.long)
        self._action_history_buf = torch.cat([self._action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # clamp delay to available history length
        history_len = self._action_history_buf.shape[1]
        delay_val = int(self.delay.item()) if torch.is_tensor(self.delay) else int(self.delay)
        if history_len <= 1:
            delay_val = 0
        else:
            delay_val = max(0, min(delay_val, history_len - 1))
        indices = -1 - delay_val
        if self._use_delay:
            self._raw_actions[:] = self._action_history_buf[:, indices]
        else:
            self._raw_actions[:] = actions
        # apply the affine transformations

        if self.cfg.clip is not None:
            self._raw_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._action_history_buf[env_ids, :, :] = 0.

    @property
    def action_history_buf(self):
        return self._action_history_buf
