"""Success reports and recording switches through the real rollout/wrapper loop."""

import json
import shutil
from types import SimpleNamespace

from gr00t.eval import rollout_policy, rollout_policy_x_emb
from gr00t.eval.sim import robocasa_x_emb
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper
import gymnasium as gym
import numpy as np
import pytest
import tyro


ENV_NAME = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"


class TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    key: gym.spaces.Box(0, 255, (32, 32, 3), np.uint8)
                    for key in rollout_policy.ROBOCASA_PANDA_RECORD_VIDEO_KEYS
                },
                "state.test": gym.spaces.Box(-1, 1, (1,), np.float32),
                "annotation.test": gym.spaces.Text(20),
            }
        )
        self.action_space = gym.spaces.Dict({"action.test": gym.spaces.Box(-1, 1, (1,))})
        self.episode = -1
        self.closed = False

    def _observation(self):
        return {
            key: "test" if key == "annotation.test" else np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get("seed"))
        self.episode += 1
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 2
        return self._observation(), 1.0, done, False, {"success": done and self.episode % 2 == 0}

    def close(self):
        self.closed = True


class TinyPolicy:
    def get_modality_config(self):
        return {key: SimpleNamespace(delta_indices=[0]) for key in ("video", "state", "action")}

    def reset(self):
        pass

    def get_action(self, observations):
        batch_size = len(observations["state.test"])
        return {"action.test": np.zeros((batch_size, 1, 1), np.float32)}, {}


@pytest.mark.parametrize("no_video", [False, True])
@pytest.mark.parametrize("x_emb", [None, {"link2": 1.2}])
def test_rollout_saves_metrics_after_close(monkeypatch, tmp_path, no_video, x_emb):
    if not no_video and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    env = TinyEnv()
    configured = []
    monkeypatch.setattr(rollout_policy, "get_gym_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(rollout_policy, "create_gr00t_sim_policy", lambda *a, **kw: TinyPolicy())
    monkeypatch.setattr(
        robocasa_x_emb, "configure_x_emb", lambda env, scales: configured.append(scales)
    )
    if no_video:

        def forbid_video(*args, **kwargs):
            pytest.fail("Video wrapper must not be created with --no_video")

        monkeypatch.setattr(VideoRecordingWrapper, "__init__", forbid_video)

    save_results = rollout_policy.save_rollout_results

    def save_after_close(*args, **kwargs):
        assert env.closed
        return save_results(*args, **kwargs)

    monkeypatch.setattr(rollout_policy, "save_rollout_results", save_after_close)
    rollout_policy.run_gr00t_sim_policy(
        ENV_NAME,
        n_episodes=2,
        max_episode_steps=2,
        n_envs=1,
        n_action_steps=1,
        video_dir=str(tmp_path),
        no_video=no_video,
        x_emb=x_emb,
        seed=0,
    )
    reports = list(tmp_path.glob("results_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["n_episodes"] == 2
    assert report["n_successes"] == 1
    assert report["success_rate"] == 0.5
    assert report["episode_successes"] == [True, False]
    assert report["episode_infos"]["episode_lengths"] == [2, 2]
    assert report["config"]["x_emb"] == (x_emb or {})
    assert report["config"]["no_video"] is no_video
    assert report["config"]["seed"] == 0
    assert configured == ([] if x_emb is None else [x_emb])
    videos = list(tmp_path.glob("*.mp4"))
    assert len(videos) == (0 if no_video else 2)
    assert all(video.stat().st_size > 0 for video in videos)


def test_reports_handle_numpy_values_and_preserve_previous_runs(tmp_path):
    results = (
        ENV_NAME,
        [np.bool_(True), np.bool_(False)],
        {"episode_lengths": [np.int64(2), np.int64(3)], "q_score": np.array([0.4, 0.8])},
    )
    first = rollout_policy.save_rollout_results(results, str(tmp_path), {})
    second = rollout_policy.save_rollout_results(results, str(tmp_path), {})
    assert first != second
    assert first.exists() and second.exists()
    assert json.loads(first.read_text())["episode_infos"]["q_score"] == [0.4, 0.8]


@pytest.mark.parametrize(
    "config_type", [rollout_policy.RolloutConfig, rollout_policy_x_emb.XEmbRolloutConfig]
)
@pytest.mark.parametrize("flag", ["--no_video", "--no-video"])
def test_no_video_cli_aliases(config_type, flag):
    assert tyro.cli(config_type, args=[flag]).no_video is True


def test_x_emb_cli_forwards_json_and_no_video(monkeypatch):
    captured = []
    args = tyro.cli(
        rollout_policy_x_emb.XEmbRolloutConfig,
        args=["--x_emb", '{"link2": 1.2}', "--no_video"],
    )
    monkeypatch.setattr(rollout_policy_x_emb.tyro, "cli", lambda *a: args)
    monkeypatch.setattr(
        rollout_policy_x_emb, "run_rollout", lambda cfg, **kw: captured.append((cfg, kw))
    )
    rollout_policy_x_emb.main()
    assert captured == [(args, {"x_emb": {"link2": 1.2}})]
    assert args.no_video
