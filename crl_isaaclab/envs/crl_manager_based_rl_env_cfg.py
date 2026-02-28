# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from crl_isaaclab.envs.crl_ui import CRLManagerBasedRLEnvWindow


@configclass
class CRLManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    ui_window_class_type: type | None = CRLManagerBasedRLEnvWindow
    crl_events: object | None = None
    costs: object | None = None
