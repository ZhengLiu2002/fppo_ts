# Config package for Galileo safe locomotion tasks.

import gymnasium as gym

from . import crl_teacher_cfg, crl_student_cfg
from crl_tasks.crl_task.config.galileo import agents


gym.register(
    id="Isaac-Galileo-CRL-Teacher-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{crl_teacher_cfg.__name__}:GalileoTeacherCRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:GalileoCRLTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Galileo-CRL-Teacher-Play-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{crl_teacher_cfg.__name__}:GalileoTeacherCRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:GalileoCRLTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Galileo-CRL-Student-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{crl_student_cfg.__name__}:GalileoStudentCRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:GalileoCRLStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Galileo-CRL-Student-Play-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{crl_student_cfg.__name__}:GalileoStudentCRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:GalileoCRLStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)
