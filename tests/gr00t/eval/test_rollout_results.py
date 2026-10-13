"""Success reports and recording switches through the real rollout/wrapper loop."""

import json
import shutil
import sys
from types import ModuleType, SimpleNamespace

from gr00t.eval import rollout_policy, rollout_policy_x_emb
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval._ik_metrics import IKDiagnostics
from gr00t.eval.sim import robocasa_x_emb
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
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
            key: "test"
            if key.startswith("annotation.")
            else np.zeros(space.shape, dtype=space.dtype)
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
    def __init__(self, expected_state=0):
        self.expected_state = expected_state

    def get_modality_config(self):
        return {key: SimpleNamespace(delta_indices=[0]) for key in ("video", "state", "action")}

    def reset(self):
        pass

    def get_action(self, observations):
        np.testing.assert_array_equal(observations["state.test"], self.expected_state)
        batch_size = len(observations["state.test"])
        return {"action.test": np.zeros((batch_size, 1, 1), np.float32)}, {}


@pytest.mark.parametrize("no_video", [False, True])
@pytest.mark.parametrize("x_emb", [None, {"link2": 1.2}])
@pytest.mark.parametrize("use_exp_adap", [False, True])
@pytest.mark.parametrize("sync_init", [False, True])
def test_rollout_saves_metrics_after_close(
    monkeypatch, tmp_path, no_video, x_emb, use_exp_adap, sync_init
):
    if not no_video and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    env = TinyEnv()
    configured = []
    adapted = []
    synced = []

    class TestSync(gym.Wrapper):
        def __init__(self, raw_env, scales):
            assert raw_env is env  # Synchronization precedes video and adaptation wrappers.
            super().__init__(raw_env)
            synced.append(scales)

    sync_module = ModuleType("gr00t.eval.sim.robocasa_sync_init")
    sync_module.SyncInitialPoseWrapper = TestSync
    monkeypatch.setitem(sys.modules, sync_module.__name__, sync_module)

    class TestAdapter(gym.ObservationWrapper):
        def __init__(self, env, scales):
            super().__init__(env)
            adapted.append(scales)

        def observation(self, obs):
            assert obs["state.test"].shape == (1,)  # Before temporal stacking / vector batching.
            return {**obs, "state.test": np.full((1,), 0.5, dtype=np.float32)}

    adapter_module = ModuleType("gr00t.eval.sim.robocasa_exp_adaptation")
    adapter_module.ExpertAdaptationWrapper = TestAdapter
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    monkeypatch.setattr(rollout_policy, "get_gym_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        rollout_policy,
        "create_gr00t_sim_policy",
        lambda *a, **kw: TinyPolicy(expected_state=0.5 if use_exp_adap else 0),
    )
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
        use_exp_adap=use_exp_adap,
        sync_init=sync_init,
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
    assert report["config"]["use_exp_adap"] is use_exp_adap
    assert report["config"]["sync_init"] is sync_init
    assert report["config"]["seed"] == 0
    assert configured == ([] if x_emb is None or sync_init else [x_emb])
    assert adapted == ([x_emb or {}] if use_exp_adap else [])
    assert synced == ([x_emb or {}] if sync_init else [])
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


@pytest.mark.parametrize("use_exp_adap", [False, True])
@pytest.mark.parametrize("sync_init", [False, True])
@pytest.mark.parametrize("sync_flag", ["--sync_init", "--sync-init"])
def test_x_emb_cli_forwards_json_and_no_video(monkeypatch, use_exp_adap, sync_init, sync_flag):
    captured = []
    args = tyro.cli(
        rollout_policy_x_emb.XEmbRolloutConfig,
        args=[
            "--x_emb",
            '{"link2": 1.2}',
            "--no_video",
            f"--use_exp_adap={use_exp_adap}",
            f"{sync_flag}={sync_init}",
        ],
    )
    monkeypatch.setattr(rollout_policy_x_emb.tyro, "cli", lambda *a: args)
    monkeypatch.setattr(
        rollout_policy_x_emb, "run_rollout", lambda cfg, **kw: captured.append((cfg, kw))
    )
    rollout_policy_x_emb.main()
    assert captured == [
        (args, {"x_emb": {"link2": 1.2}, "use_exp_adap": use_exp_adap, "sync_init": sync_init})
    ]
    assert args.no_video


def test_shared_main_forwards_expert_adaptation_and_sync_init(monkeypatch):
    captured = {}

    def capture_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rollout_policy, "run_gr00t_sim_policy", capture_run)
    args = rollout_policy_x_emb.XEmbRolloutConfig(
        policy_client_host="127.0.0.1", policy_client_port=5555
    )
    rollout_policy.main(args, x_emb={"link2": 1.2}, use_exp_adap=True, sync_init=True)
    assert captured["use_exp_adap"] is True
    assert captured["sync_init"] is True
    assert captured["x_emb"] == {"link2": 1.2}


class DiagnosticTinyEnv(TinyEnv):
    """A reset failure and a mid-chunk peak, followed by an all-converged episode."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.diagnostics = IKDiagnostics()
        bad_reset = self.episode % 2 == 0
        info.update(
            self.diagnostics.record(
                0.01 if bad_reset else 0, 0.02 if bad_reset else 0, not bad_reset
            )
        )
        return obs, info

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)
        position, rotation = (0.0, 0.0)
        if self.episode % 2 == 0:
            position, rotation = [(0.002, 0.003), (0, 0), (0.004, 0.04), (0, 0), (0, 0)][
                self.steps - 1
            ]
        info.update(
            self.diagnostics.record(position, rotation, position <= 1e-4 and rotation <= 1e-3)
        )
        return obs, reward, self.steps == 5, truncated, info


class ChunkPolicy(TinyPolicy):
    def get_action(self, observations):
        batch_size = len(observations["state.test"])
        return {"action.test": np.zeros((batch_size, 3, 1), np.float32)}, {}


@pytest.mark.parametrize("n_envs", [1, 2])
def test_ik_report_preserves_reset_and_mid_chunk_peaks_across_episodes(tmp_path, n_envs, capsys):
    contract = PolicyHorizonSpec(3, 3, (0,), (0,))
    env = gym.vector.SyncVectorEnv(
        [
            lambda: MultiStepWrapper(
                DiagnosticTinyEnv(),
                contract=contract,
                max_episode_steps=5,
                terminate_on_success=False,
            )
        ]
        * n_envs
    )
    try:
        successes, lengths, _, infos = rollout_policy._collect_rollout_episodes(
            env, ChunkPolicy(), n_episodes=2 * n_envs, n_envs=n_envs, seed=0
        )
    finally:
        env.close()
    assert lengths == [5] * (2 * n_envs)
    metrics = infos["exp_adap_ik"]
    assert len(metrics) == 2 * n_envs
    for index, item in enumerate(metrics):
        first_episode = index < n_envs
        assert item["exp_adap_ik_samples"] == 6  # reset + 5 steps, final chunk only 2 steps
        assert item["exp_adap_ik_failures"] == (3 if first_episode else 0)
        assert item["exp_adap_ik_failure_rate"] == (0.5 if first_episode else 0)
        assert item["exp_adap_ik_max_position_error_m"] == (0.01 if first_episode else 0)
        assert item["exp_adap_ik_max_rotation_error_rad"] == (0.04 if first_episode else 0)
    report_path = rollout_policy.save_rollout_results(
        (ENV_NAME, successes, infos), str(tmp_path), {}
    )
    assert json.loads(report_path.read_text())["episode_infos"]["exp_adap_ik"] == metrics
    output = capsys.readouterr().out
    assert "max_pos=10.000 mm" in output
    assert "max_rot=2.292 deg" in output
    assert "outside_tolerance=3 (50.00%)" in output


def test_gui_prints_episode_ik_statistics(capsys):
    from scripts.eval_robocasa_gui import run_episodes

    contract = PolicyHorizonSpec(3, 3, (0,), (0,))
    raw = DiagnosticTinyEnv()
    raw.observation_space["annotation.human.action.task_description"] = gym.spaces.Text(20)
    env = MultiStepWrapper(raw, contract=contract, max_episode_steps=5, terminate_on_success=False)
    gui = SimpleNamespace(
        viewer=SimpleNamespace(is_running=lambda: True), finish_recording=lambda: None
    )
    try:
        run_episodes(env, gui, ChunkPolicy(), n_episodes=2, seed=0)
    finally:
        env.close()
    output = capsys.readouterr().out
    assert output.count("IK (reset + steps): samples=6") == 2
    assert "outside_tolerance=3 (50.00%)" in output
    assert "outside_tolerance=0 (0.00%)" in output
