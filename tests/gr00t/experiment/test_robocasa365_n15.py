"""Exercise the N1.5 wire bridge and kitchen rollout without model weights/assets."""

from contextlib import nullcontext
import json
import queue
import sys
import threading
from types import ModuleType, SimpleNamespace

from gr00t.experiment.custom import (
    robocasa365_n15 as kitchen,
    run_gr00t_on_robocas as experiment,
    serve_robocasa365_n15 as bridge,
)
from gr00t.policy.server_client import MsgSerializer, PolicyClient
import gymnasium as gym
import msgpack
import numpy as np
import pytest
import zmq


class FakeN15Policy:
    def __init__(self):
        self.requests = []

    def get_modality_config(self):
        return {
            name: SimpleNamespace(
                delta_indices=config.delta_indices,
                modality_keys=[
                    key if name == "language" else f"{name}.{key}" for key in config.modality_keys
                ],
            )
            for name, config in kitchen.modality_config().items()
        }

    def get_action(self, obs):
        self.requests.append(obs)
        return {
            f"action.{key}": np.full((1, 16, dim), 0.25, dtype=np.float32)
            for key, dim in kitchen.ACTION_DIMS.items()
        }


class FakeKitchen(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"state.{key}": gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                    for key, dim in kitchen.STATE_DIMS.items()
                },
                **{
                    f"video.{key}": gym.spaces.Box(0, 255, (2, 3, 3), np.uint8)
                    for key in kitchen.CAMERA_NAMES
                },
                kitchen.LANGUAGE_KEY: gym.spaces.Text(256),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                f"action.{key}": gym.spaces.Box(-1, 1, (dim,), np.float32)
                for key, dim in kitchen.ACTION_DIMS.items()
            }
        )
        self.actions = []
        self.closed = False
        self.seed = None

    def observation(self):
        return {
            **{
                f"state.{key}": np.zeros(dim, dtype=np.float32)
                for key, dim in kitchen.STATE_DIMS.items()
            },
            **{
                f"video.{key}": np.full((2, 3, 3), i + len(self.actions), dtype=np.uint8)
                for i, key in enumerate(kitchen.CAMERA_NAMES)
            },
            kitchen.LANGUAGE_KEY: "open the cabinet",
        }

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        return self.observation(), {"success": False}

    def step(self, action):
        self.actions.append(action)
        return self.observation(), 0.0, False, False, {"success": False}

    def close(self):
        self.closed = True


def test_bridge_communicates_with_real_policy_client():
    policy = FakeN15Policy()
    adapter = bridge.N15PolicyBridge(policy)
    ports = queue.Queue()
    errors = []

    def run_server():
        try:
            with zmq.Context() as context, context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.setsockopt(zmq.LINGER, 0)
                ports.put(socket.bind_to_random_port("tcp://127.0.0.1"))
                for _ in range(4):
                    request = bridge.unpack(socket.recv())
                    socket.send(bridge.pack(adapter.dispatch(request)))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    try:
        with PolicyClient(port=ports.get(timeout=5), timeout_ms=5000) as client:
            assert client.ping()
            kitchen.validate_modality_config(client.get_modality_config())
            assert client.reset() == {}
            raw = FakeKitchen().observation()
            obs = {
                key: [value] if isinstance(value, str) else value[None, None]
                for key, value in raw.items()
            }
            actions, info = client.get_action(obs)
            assert info == {}
            assert set(actions) == {f"action.{key}" for key in kitchen.ACTION_DIMS}
            np.testing.assert_array_equal(actions["action.end_effector_rotation"], 0.25)
    finally:
        thread.join(timeout=6)
    assert not thread.is_alive()
    assert not errors
    received = policy.requests[0]
    assert received[kitchen.LANGUAGE_KEY].tolist() == [["open the cabinet"]]
    for camera in kitchen.CAMERA_NAMES:
        np.testing.assert_array_equal(received[f"video.{camera}"], obs[f"video.{camera}"])


def test_bridge_rejects_object_arrays():
    with pytest.raises(ValueError):
        bridge.pack(np.array([object()], dtype=object))
    payload = msgpack.packb({b"nd": True, b"kind": b"O", b"data": b"unused"})
    with pytest.raises(ValueError, match="Object arrays"):
        bridge.unpack(payload)


def test_kitchen_rollout_shows_exact_policy_inputs_and_executes_partial_chunk(monkeypatch, capsys):
    raw_env = FakeKitchen()
    adapter = kitchen.InteractiveKitchenEnv(raw_env, visualize=False, instruction="close it")
    monkeypatch.setattr(kitchen, "create_env", lambda _: adapter)
    snapshots = []
    viewer = SimpleNamespace(polls=0, closed=False)
    viewer.update = lambda instruction, frames, **kw: snapshots.append((instruction, frames, kw))
    viewer.poll = lambda: setattr(viewer, "polls", viewer.polls + 1)
    viewer.close = lambda: setattr(viewer, "closed", True)
    monkeypatch.setattr(experiment, "PolicyInputViewer", lambda *args: viewer)
    requests = []

    def get_action(obs):
        instruction, frames, metadata = snapshots[-1]
        assert instruction == obs[kitchen.LANGUAGE_KEY][0] == "close it"
        for camera in kitchen.CAMERA_NAMES:
            np.testing.assert_array_equal(frames[camera], obs[f"video.{camera}"][0])
        requests.append(metadata["step"])
        return FakeN15Policy().get_action(obs), {}

    policy = SimpleNamespace(
        get_modality_config=kitchen.modality_config, reset=lambda: None, get_action=get_action
    )
    from gr00t.policy import server_client

    monkeypatch.setattr(server_client, "PolicyClient", lambda **kw: nullcontext(policy))
    experiment.main(experiment.Config(mode="robocasa365-n15", max_episode_steps=21, seed=7))
    assert requests == [0, 16]
    assert len(raw_env.actions) == 21
    assert raw_env.seed == 7
    assert raw_env.closed and viewer.closed
    assert viewer.polls == 21
    # Kitchen actions retain the official normalized OSC command scale.
    for action in raw_env.actions:
        np.testing.assert_array_equal(action["action.end_effector_rotation"], 0.25)
    summary = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert summary["task"] == "OpenCabinet"
    assert summary["robot"] == "PandaOmron"
    assert summary["n_action_steps"] == 16
    assert summary["split"] == "target"


def test_smoke_test_uses_zero_deltas_and_never_connects(monkeypatch, capsys):
    raw_env = FakeKitchen()
    adapter = kitchen.InteractiveKitchenEnv(raw_env, visualize=False, instruction=None)
    monkeypatch.setattr(kitchen, "create_env", lambda _: adapter)
    from gr00t.policy import server_client

    monkeypatch.setattr(server_client, "PolicyClient", lambda **kw: pytest.fail("contacted server"))
    experiment.main(
        experiment.Config(
            mode="robocasa365-n15", max_episode_steps=2, smoke_test=True, visualize=False
        )
    )
    assert len(raw_env.actions) == 2
    for action in raw_env.actions:
        for value in action.values():
            np.testing.assert_array_equal(value, 0.0)
    assert "success_rate" not in capsys.readouterr().out


def test_invalid_action_rejected_before_stepping(monkeypatch):
    raw_env = FakeKitchen()
    adapter = kitchen.InteractiveKitchenEnv(raw_env, visualize=False, instruction=None)
    monkeypatch.setattr(kitchen, "create_env", lambda _: adapter)
    actions = FakeN15Policy().get_action({})
    actions["action.end_effector_rotation"][0, 0, 0] = np.nan
    policy = SimpleNamespace(
        get_modality_config=kitchen.modality_config,
        reset=lambda: None,
        get_action=lambda _: (actions, {}),
    )
    from gr00t.policy import server_client

    monkeypatch.setattr(server_client, "PolicyClient", lambda **kw: nullcontext(policy))
    with pytest.raises(ValueError, match="Invalid action.end_effector_rotation"):
        experiment.main(
            experiment.Config(mode="robocasa365-n15", max_episode_steps=2, visualize=False)
        )
    assert not raw_env.actions
    assert raw_env.closed


def test_wrong_server_schema_is_rejected():
    wrong = kitchen.modality_config()
    wrong["video"].modality_keys = ["ego_view"]
    with pytest.raises(ValueError, match="serve_robocasa365_n15"):
        kitchen.validate_modality_config(wrong)


def test_default_g1_configuration_preserved():
    config = experiment.resolve_config(experiment.Config())
    assert (config.task, config.max_episode_steps, config.n_action_steps) == (
        "PnPAppleToPlate",
        400,
        8,
    )
    assert config.policy_port == 5555


def test_kitchen_defaults_use_task_horizon_and_allow_overrides(monkeypatch):
    registry = ModuleType("robocasa.utils.dataset_registry_utils")
    tasks = []
    registry.get_task_horizon = lambda task: tasks.append(task) or 1200
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(kitchen, "load_robocasa365", lambda _: None)
    config = experiment.resolve_config(experiment.Config(mode="robocasa365-n15"))
    assert (config.task, config.max_episode_steps, config.n_action_steps, config.policy_port) == (
        "OpenCabinet",
        1200,
        16,
        5556,
    )
    assert tasks == ["OpenCabinet"]
    override = experiment.resolve_config(
        experiment.Config(mode="robocasa365-n15", max_episode_steps=80, n_action_steps=4)
    )
    assert (override.max_episode_steps, override.n_action_steps) == (80, 4)
    assert tasks == ["OpenCabinet"]


def test_kitchen_factory_uses_official_wrapper_and_native_camera_resolution(monkeypatch):
    module = ModuleType("robocasa.wrappers.gym_wrapper")
    registry = ModuleType("robosuite.environments.base")
    registry.REGISTERED_ENVS = {"OpenCabinet": object}
    calls = []
    raw_env = FakeKitchen()
    module.RoboCasaGymEnv = lambda **kw: calls.append(kw) or raw_env
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(kitchen, "load_robocasa365", lambda _: None)
    config = experiment.resolve_config(
        experiment.Config(mode="robocasa365-n15", max_episode_steps=100, visualize=False)
    )
    env = kitchen.create_env(config)
    try:
        assert calls == [
            {
                "env_name": "OpenCabinet",
                "split": "target",
                "enable_render": True,
                "camera_widths": 256,
                "camera_heights": 256,
                "seed": 0,
                "horizon": 100,
            }
        ]
        obs, _ = env.reset()
        assert set(obs) == set(raw_env.observation())
    finally:
        env.close()
    assert raw_env.closed


def test_wire_preserves_numpy_arrays_in_both_directions():
    value = {"pixels": np.arange(18, dtype=np.uint8).reshape(1, 2, 3, 3)}
    np.testing.assert_array_equal(
        bridge.unpack(MsgSerializer.to_bytes(value))["pixels"], value["pixels"]
    )
    np.testing.assert_array_equal(
        MsgSerializer.from_bytes(bridge.pack(value))["pixels"], value["pixels"]
    )
