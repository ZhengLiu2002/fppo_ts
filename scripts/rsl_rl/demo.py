"""
Code reference:
1. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/enum_namespacecarb_1_1input_1a41f626f5bfc1020c9bd87f5726afdec1.html#namespacecarb_1_1input_1a41f626f5bfc1020c9bd87f5726afdec1
2. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/api/enum_namespacecarb_1_1input_1af1c4ed7e318b3719809f13e2a48e2f2d.html#namespacecarb_1_1input_1af1c4ed7e318b3719809f13e2a48e2f2d
3. https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/docs/python/bindings.html#carb.input.GamepadInput
"""

import argparse
import os
import sys
import weakref
from pathlib import Path

# Ensure repository root is on sys.path for `scripts.*` imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TASKS_ROOT = _REPO_ROOT / "crl_tasks"
if _TASKS_ROOT.is_dir() and str(_TASKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASKS_ROOT))

import cli_args  # isort: skip
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length", type=int, default=500, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time", action="store_true", default=False, help="Run in real-time, if possible."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from crl_isaaclab.envs import CRLManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from scripts.rsl_rl.vecenv_wrapper import CRLRslRlVecEnvWrapper
from crl_tasks.tasks.galileo.config.agents.rsl_rl_cfg import CRLRslRlOnPolicyRunnerCfg
from crl_tasks.tasks.galileo.config.teacher_env_cfg import GalileoTeacherCRLEnvCfg_PLAY
from crl_tasks.tasks.galileo.config.student_env_cfg import GalileoStudentCRLEnvCfg_PLAY


class CRLDemoGalileo:
    def __init__(self):
        agent_cfg: CRLRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        if args_cli.use_pretrained_checkpoint:
            checkpoint = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
            if not checkpoint:
                print(
                    "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
                )
                return
        elif args_cli.checkpoint:
            checkpoint = retrieve_file_path(args_cli.checkpoint)
        else:
            checkpoint = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )

        self.agent_cfg = agent_cfg
        # create envionrment
        env_cfg = (
            GalileoTeacherCRLEnvCfg_PLAY()
            if "Teacher" in args_cli.task
            else GalileoStudentCRLEnvCfg_PLAY()
        )
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        self.env_cfg = env_cfg
        # wrap around environment for rsl-rl
        self.env = CRLRslRlVecEnvWrapper(CRLManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunnerWithExtractor(
            self.env, agent_cfg.to_dict(), log_dir=None, device=self.device
        )
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")
        # self.set_up_keyboard()
        self.set_up_gamepad()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        # self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self._camera_local_transform = torch.tensor([-0.0, 2.6, 1.6], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_gamepad(self):
        self._input = carb.input.acquire_input_interface()
        self._gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self.dead_zone = 0.01
        self.v_x_sensitivity = 0.8
        self.v_y_sensitivity = 0.8
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: self.env_cfg.commands.base_velocity.ranges.lin_vel_x[
                1
            ],
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: self.env_cfg.commands.base_velocity.ranges.lin_vel_x[
                0
            ],
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: self.env_cfg.commands.base_velocity.ranges.heading[
                0
            ],
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: self.env_cfg.commands.base_velocity.ranges.heading[
                1
            ],
        }

    def _on_gamepad_event(self, event):
        if event.type == carb.input.GamepadConnectionEventType.CONNECTED:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            cur_val = event.value
            if abs(cur_val) < self.dead_zone:
                cur_val = 0
            if event.input in self._INPUT_STICK_VALUE_MAPPING:
                if self._selected_id:
                    value = self._INPUT_STICK_VALUE_MAPPING[event.input.name]
                    self.commands[self._selected_id] = value * cur_val
            # Escape key exits out of the current selected robot view
            elif event.input == "LEFT_SHOULDER":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input == "RIGHT_SHOULDER":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.GamepadConnectionEventType.DISCONNECTED:
            if self._selected_id:
                self.commands[self._selected_id] = torch.zeros(1, 3).to(self.device)

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a Galileo robot")

        # Reset commands for previously selected robot if a new one is selected
        if (
            self._previous_selected_id is not None
            and self._previous_selected_id != self._selected_id
        ):
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[
            self._selected_id, :
        ]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_galileo = CRLDemoGalileo()
    obs, extras = demo_galileo.env.reset()
    while simulation_app.is_running():
        # check for selected robots
        demo_galileo.update_selected_object()
        with torch.inference_mode():

            action = demo_galileo.policy(obs, hist_encoding=True)
            obs, _, _, extras = demo_galileo.env.step(action)
            # overwrite command based on keyboard input


if __name__ == "__main__":
    main()
    simulation_app.close()
