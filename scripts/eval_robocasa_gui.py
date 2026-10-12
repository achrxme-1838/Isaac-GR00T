r"""Run a RoboCasa policy client with a GUI, or record videos without a window.

With the policy server running, copy into another terminal:

cd /home/lim/rl_ws/Isaac-GR00T
unset PYTHONPATH PYTHONHOME PYOPENGL_PLATFORM
MUJOCO_GL=egl \
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/eval_robocasa_gui.py \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --n-episodes 10 --max-episode-steps 720 --n-action-steps 8 --seed 0 \
  --record_vid

--record_vid disables the GUI and saves 960x720 MP4s to
eval_videos/gui_<timestamp>/episode_NNN.mp4 using robot0_agentview_left (arm and workspace).
The recording camera is pulled back 0.5 m to keep more of the arm in view.
Without --record_vid, the interactive GUI opens as usual.
Use --video-dir PATH or --record-camera robot0_agentview_right to override these defaults.
"""

import argparse
from pathlib import Path
import time

from gr00t.eval._ik_metrics import extract_ik_metrics, format_ik_metrics
from gr00t.eval.sim.gui_video_recorder import (
    DEFAULT_RECORD_CAMERA,
    GuiVideoRecorder,
    add_recording_args,
    recording_directory,
)
import gymnasium as gym
import numpy as np


class GuiWrapper(gym.Wrapper):
    def __init__(
        self, env, *, overlay_factory=None, video_dir=None, record_camera=DEFAULT_RECORD_CAMERA
    ):
        super().__init__(env)
        self.viewer = None
        self.overlay_factory = overlay_factory
        self.overlay = None
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.record_camera = record_camera
        self.recorder = None
        self.episode = 0

    def reset(self, **kwargs):
        # RoboCasa replaces the MuJoCo model on reset; close its old viewer first.
        self.finish_recording()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.overlay = None
        obs, info = self.env.reset(**kwargs)
        robot_env = self.unwrapped.env
        if self.overlay_factory is not None:
            self.overlay = self.overlay_factory(robot_env)
        if self.video_dir is None:
            import mujoco.viewer

            sim = robot_env.sim
            self.viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
            with self.viewer.lock():
                # Match RoboSuite camera rendering: hide collision geometry by default.
                self.viewer.opt.geomgroup[0] = int(robot_env.render_collision_mesh)
                self.viewer.opt.geomgroup[1] = int(robot_env.render_visual_mesh)
                for key, value in robot_env.renderer_config["cam_config"].items():
                    setattr(self.viewer.cam, key, value)
                if self.overlay is not None:
                    self.overlay.update(self.viewer.user_scn, obs["state.joint_position"])
            self.viewer.sync()
        self.episode += 1
        if self.video_dir is not None:
            self.recorder = GuiVideoRecorder(
                robot_env,
                self.video_dir / f"episode_{self.episode:03d}.mp4",
                camera=self.record_camera,
                overlay=self.overlay,
            )
            self.recorder.write(obs)
        return obs, info

    def step(self, action):
        start = time.perf_counter()
        if self.viewer is None:
            result = self.env.step(action)
        else:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            with self.viewer.lock():
                result = self.env.step(action)
                if self.overlay is not None:
                    self.overlay.update(self.viewer.user_scn, result[0]["state.joint_position"])
            self.viewer.sync()
        if self.recorder is not None:
            self.recorder.write(result[0])
        if self.viewer is not None:
            time.sleep(max(0, 1 / self.unwrapped.env.control_freq - (time.perf_counter() - start)))
        return result

    def finish_recording(self):
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            recorder.close()

    def close(self):
        try:
            self.finish_recording()
        finally:
            try:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                self.overlay = None
            finally:
                self.env.close()


def run_episodes(env, gui, policy, n_episodes, seed):
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        policy.reset()
        print(
            f"Episode {episode + 1}: {obs['annotation.human.action.task_description']}", flush=True
        )
        steps = 0
        while gui.viewer is None or gui.viewer.is_running():
            batched_obs = {
                key: [value] if isinstance(value, str) else np.expand_dims(value, 0)
                for key, value in obs.items()
            }
            actions, _ = policy.get_action(batched_obs)
            obs, _, terminated, truncated, info = env.step(
                {key: value[0] for key, value in actions.items()}
            )
            steps += info["n_env_steps"]
            if terminated or truncated:
                print(f"  Steps: {steps}, success: {bool(np.any(info['success']))}", flush=True)
                ik_metrics = extract_ik_metrics(info)
                if ik_metrics:
                    print(f"  {format_ik_metrics(ik_metrics)}", flush=True)
                break
        gui.finish_recording()
        if gui.viewer is not None and not gui.viewer.is_running():
            break


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-name", default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    add_recording_args(parser)
    args = parser.parse_args(argv)
    if min(args.n_episodes, args.max_episode_steps, args.n_action_steps) < 1:
        parser.error("Episode and step counts must be positive.")
    return args


def main():
    args = parse_args()

    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from gr00t.policy.server_client import PolicyClient
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    with PolicyClient(
        host=args.policy_client_host, port=args.policy_client_port, timeout_ms=120000
    ) as policy:
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
        gui = GuiWrapper(
            gym.make(args.env_name, enable_render=True, seed=args.seed),
            video_dir=recording_directory(args),
            record_camera=args.record_camera,
        )
        try:
            env = MultiStepWrapper(
                gui,
                contract=contract,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            )
            run_episodes(env, gui, policy, args.n_episodes, args.seed)
        except KeyboardInterrupt:
            print("Evaluation stopped.")
        finally:
            gui.close()


if __name__ == "__main__":
    main()
