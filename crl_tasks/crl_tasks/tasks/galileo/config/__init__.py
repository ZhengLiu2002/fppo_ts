"""Configuration package for Galileo CRL tasks."""

from . import agents
from .defaults import GalileoDefaults
from .mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    StudentCostsCfg,
    StudentObservationsCfg,
    StudentRewardsCfg,
    TeacherCostsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)
from .student_env_cfg import (
    GalileoStudentCRLEnvCfg,
    GalileoStudentCRLEnvCfg_EVAL,
    GalileoStudentCRLEnvCfg_PLAY,
)
from .teacher_env_cfg import (
    GalileoCRLSceneCfg,
    GalileoTeacherCRLEnvCfg,
    GalileoTeacherCRLEnvCfg_PLAY,
)

__all__ = [
    "agents",
    "GalileoDefaults",
    "ActionsCfg",
    "CommandsCfg",
    "CurriculumCfg",
    "EventCfg",
    "StudentCostsCfg",
    "StudentObservationsCfg",
    "StudentRewardsCfg",
    "TeacherCostsCfg",
    "TeacherObservationsCfg",
    "TeacherRewardsCfg",
    "TerminationsCfg",
    "GalileoCRLSceneCfg",
    "GalileoTeacherCRLEnvCfg",
    "GalileoTeacherCRLEnvCfg_PLAY",
    "GalileoStudentCRLEnvCfg",
    "GalileoStudentCRLEnvCfg_EVAL",
    "GalileoStudentCRLEnvCfg_PLAY",
]
