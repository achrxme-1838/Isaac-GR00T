"""Run a RoboCasa policy client with an interactive MuJoCo viewer, without recording."""

import argparse
import time

import gymnasium as gym
import numpy as np


class GuiWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewer = None

    def reset(self, **kwargs):
        import mujoco.viewer

        # RoboCasa replaces the MuJoCo model on reset; close its old viewer first.
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        obs, info = self.env.reset(**kwargs)
        robot_env = self.unwrapped.env
        sim = robot_env.sim
        self.viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
        with self.viewer.lock():
            # Match RoboSuite camera rendering: hide collision geometry by default.
            self.viewer.opt.geomgroup[0] = int(robot_env.render_collision_mesh)
            self.viewer.opt.geomgroup[1] = int(robot_env.render_visual_mesh)
            for key, value in robot_env.renderer_config["cam_config"].items():
                setattr(self.viewer.cam, key, value)
        self.viewer.sync()
        return obs, info

    def step(self, action):
        if not self.viewer.is_running():
            raise KeyboardInterrupt
        start = time.perf_counter()
        with self.viewer.lock():
            result = self.env.step(action)
        self.viewer.sync()
        time.sleep(max(0, 1 / self.unwrapped.env.control_freq - (time.perf_counter() - start)))
        return result

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.env.close()


def run_episodes(env, gui, policy, n_episodes, seed):
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        policy.reset()
        print(
            f"Episode {episode + 1}: {obs['annotation.human.action.task_description']}", flush=True
        )
        steps = 0
        while gui.viewer.is_running():
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
                break
        if not gui.viewer.is_running():
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.n_episodes, args.max_episode_steps, args.n_action_steps) < 1:
        parser.error("Episode and step counts must be positive.")

    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from gr00t.policy.server_client import PolicyClient
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    with PolicyClient(
        host=args.policy_client_host, port=args.policy_client_port, timeout_ms=120000
    ) as policy:
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
        gui = GuiWrapper(gym.make(args.env_name, enable_render=True, seed=args.seed))
        try:
            env = MultiStepWrapper(
                gui,
                contract=contract,
                max_episode_steps=args.max_episode_steps,
                terminate_on_success=True,
            )
            run_episodes(env, gui, policy, args.n_episodes, args.seed)
        except KeyboardInterrupt:
            print("Viewer stopped.")
        finally:
            gui.close()


if __name__ == "__main__":
    main()
