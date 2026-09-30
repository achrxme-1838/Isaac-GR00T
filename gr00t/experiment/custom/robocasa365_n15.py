"""RoboCasa365 N1.5 rollout schema and interactive simulator adapter."""

from pathlib import Path
import sys
import time

from gr00t.data.types import ModalityConfig
import gymnasium as gym
import numpy as np


CAMERA_NAMES = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
LANGUAGE_KEY = "annotation.human.task_description"
STATE_DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}
ACTION_DIMS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}


def modality_config():
    """Only used by smoke tests; policy runs query the actual server."""
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(CAMERA_NAMES)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_DIMS)),
        "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=list(ACTION_DIMS)),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }


def validate_modality_config(modalities):
    for name, expected in modality_config().items():
        actual = modalities.get(name)
        if actual is None or set(actual.modality_keys) != set(expected.modality_keys):
            raise ValueError(
                f"Expected RoboCasa365 N1.5 panda_omron {name} keys "
                f"{expected.modality_keys}. Start serve_robocasa365_n15.py in the N1.5 environment."
            )


def load_robocasa365(repo_path: Path):
    repo_path = repo_path.expanduser().resolve()
    if not (repo_path / "robocasa/utils/dataset_registry_utils.py").is_file():
        raise FileNotFoundError(
            f"RoboCasa365 checkout missing at {repo_path}. Run "
            "bash gr00t/eval/sim/robocasa365/setup_RoboCasa365.sh and use its Python, "
            "or set --robocasa365-path to an installed RoboCasa365 checkout."
        )
    sys.path.insert(0, str(repo_path))
    import robocasa

    if not Path(robocasa.__file__).resolve().is_relative_to(repo_path):
        raise RuntimeError(f"A different RoboCasa is already imported: {robocasa.__file__}")


class InteractiveKitchenEnv(gym.Wrapper):
    """Viewer shares simulation state; offscreen policy cameras remain enabled."""

    def __init__(self, env, *, visualize: bool, instruction: str | None):
        super().__init__(env)
        self.visualize = visualize
        self.instruction = instruction
        self.viewer = None
        self.viewer_closed = False
        self.input_viewer = None

    def observation(self, obs):
        result = {f"state.{key}": obs[f"state.{key}"] for key in STATE_DIMS}
        result.update({f"video.{key}": obs[f"video.{key}"] for key in CAMERA_NAMES})
        result[LANGUAGE_KEY] = obs[LANGUAGE_KEY] if self.instruction is None else self.instruction
        for key, dim in STATE_DIMS.items():
            value = np.asarray(result[f"state.{key}"])
            if value.shape != (dim,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid RoboCasa365 state.{key}: expected ({dim},), got {value}")
        return result

    def reset(self, *, seed=None, options=None):
        # A hard reset may replace MuJoCo model/data; never leave the old viewer attached.
        self._close_viewer()
        obs, info = self.env.reset(seed=seed, options=options)
        self.viewer_closed = False
        if self.visualize:
            import mujoco.viewer

            sim = self.env.env.sim
            self.viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
            with self.viewer.lock():
                self.viewer.opt.geomgroup[0] = 0
                self.viewer.cam.lookat[:] = sim.data.get_site_xpos(
                    self.env.env.robots[0].gripper["right"].important_sites["grip_site"]
                )
                self.viewer.cam.distance = 3.0
                self.viewer.cam.elevation = -25
            self.viewer.sync()
        return self.observation(obs), info

    def step(self, action):
        start = time.monotonic()
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.viewer is not None:
            self.viewer_closed = not self.viewer.is_running()
            if not self.viewer_closed:
                self.viewer.sync()
            time.sleep(max(0.0, self.env.env.control_timestep - (time.monotonic() - start)))
        if self.input_viewer is not None:
            self.input_viewer.poll()
        return self.observation(obs), reward, terminated, truncated or self.viewer_closed, info

    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def close(self):
        self._close_viewer()
        self.env.close()


def create_env(config):
    load_robocasa365(config.robocasa365_path)
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from robosuite.environments.base import REGISTERED_ENVS

    if config.task not in REGISTERED_ENVS:
        raise ValueError(f"Unknown RoboCasa365 task {config.task!r}")
    # Use the benchmark's own key/action conversion and native 256px renders.
    # The separate N1.7 wrapper renders at 512px and has additional aliases.
    env = RoboCasaGymEnv(
        env_name=config.task,
        split=config.robocasa_split,
        enable_render=True,
        camera_widths=256,
        camera_heights=256,
        seed=config.seed,
        horizon=config.max_episode_steps,
    )
    return InteractiveKitchenEnv(env, visualize=config.visualize, instruction=config.instruction)


def smoke_actions(action_horizon):
    # OSC controllers consume normalized deltas. Zero holds EEF/base/torso;
    # binary zero selects the open gripper and arm control mode.
    return {
        f"action.{key}": np.zeros((action_horizon, dim), dtype=np.float32)
        for key, dim in ACTION_DIMS.items()
    }
