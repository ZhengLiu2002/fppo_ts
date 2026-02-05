# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path for `scripts.*` imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TASKS_ROOT = _REPO_ROOT / "crl_tasks"
if _TASKS_ROOT.is_dir() and str(_TASKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASKS_ROOT))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from crl_tasks.crl_task.config.galileo.agents.crl_rl_cfg import CRLRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit,
export_teacher_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def _print_terrain_summary(base_env):
    """Log terrain generator shape and current terrain level/type distribution."""
    try:
        terrain = base_env.scene.terrain
        levels = terrain.terrain_levels.cpu().numpy()
        types = terrain.terrain_types.cpu().numpy()
        uniq_lvl, lvl_counts = np.unique(levels, return_counts=True)
        uniq_typ, typ_counts = np.unique(types, return_counts=True)
        gen_cfg = terrain.cfg.terrain_generator
        print("[INFO] Terrain generator layout:"
              f" rows={getattr(gen_cfg, 'num_rows', '?')},"
              f" cols={getattr(gen_cfg, 'num_cols', '?')},"
              f" curriculum={getattr(gen_cfg, 'curriculum', '?')},"
              f" difficulty_range={getattr(gen_cfg, 'difficulty_range', '?')}")
        print("[INFO] Terrain level histogram (level:count):",
              {int(k): int(v) for k, v in zip(uniq_lvl, lvl_counts)})
        print("[INFO] Terrain type histogram (col_idx:count):",
              {int(k): int(v) for k, v in zip(uniq_typ, typ_counts)})
    except Exception as exc:  # pragma: no cover - debug aid
        print(f"[WARN] Unable to print terrain summary: {exc}")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Play 模式尽量避免超时重置，只在到达最后目标或摔倒时重置（训练不受影响）
    try:
        base_env = env.unwrapped
        if hasattr(base_env, "max_episode_length"):
            new_len = int(base_env.max_episode_length * 100)  # effectively disable time-out for play
            base_env.max_episode_length = new_len
            if hasattr(base_env, "cfg") and hasattr(base_env.cfg, "episode_length_s"):
                base_env.cfg.episode_length_s = base_env.cfg.episode_length_s * 100.0
            print(f"[INFO] Play mode: time-out relaxed to {new_len} steps (goal/fall still terminate).")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to relax play time-out: {exc}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CRLRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = ppo_runner.alg.policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_policy")
    export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_teacher_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    # reset environment
    obs, extras = env.get_observations()
    _print_terrain_summary(env.unwrapped)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs, hist_encoding=True)
        obs, _, _, extras = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
