from isaaclab.utils import configclass
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg


@configclass
class CRLManagerBasedEnvCfg(ManagerBasedEnvCfg):
    crl_events: object | None = None
