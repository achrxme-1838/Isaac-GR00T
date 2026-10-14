"""CPU coverage of periodic evaluation, simulator IPC, and distributed failures."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from gr00t.configs.base_config import Config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.rollout_eval_config import RolloutEvalConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.experiment import rollout_eval
from gr00t.experiment.rollout_eval import RolloutEvalCallback, evaluation_state
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import TrainingArguments
from transformers.integrations import WandbCallback
import tyro


TASK = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"


def _report(task=TASK):
    return {
        "env_name": task,
        "n_episodes": 2,
        "n_successes": 1,
        "success_rate": 0.5,
        "episode_infos": {"episode_rewards": [1, 3], "episode_lengths": [4, 8]},
    }


def _processor():
    modalities = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["camera"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    processor = MagicMock()
    processor.get_modality_configs.return_value = {
        EmbodimentTag.ROBOCASA_PANDA_OMRON.value: modalities
    }
    return processor


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eval_steps": -1},
        {"eval_steps": 10},
        {"eval_steps": 10, "eval_tasks": [TASK, TASK]},
        {"eval_steps": 10, "eval_tasks": [TASK], "eval_n_episodes": 0},
        {"eval_steps": 10, "eval_tasks": [TASK], "eval_n_envs": 0},
        {"eval_steps": 10, "eval_tasks": [TASK], "eval_timeout": 0},
        {"eval_steps": 10, "eval_tasks": [TASK], "eval_seed": -1},
    ],
)
def test_invalid_finetune_evaluation_options(kwargs):
    with pytest.raises(ValueError, match="eval_"):
        FinetuneConfig(base_model_path="base", dataset_path="data", embodiment_tag="tag", **kwargs)


def test_rollout_config_survives_yaml_roundtrip(tmp_path):
    config = Config()
    config.training.rollout_eval = RolloutEvalConfig(steps=5, tasks=[TASK], n_episodes=4)
    path = tmp_path / "config.yaml"
    config.save(path)
    loaded = Config().load(path)
    assert loaded.training.rollout_eval == config.training.rollout_eval
    loaded.training.rollout_eval.validate()


@pytest.mark.parametrize("quoted", [False, True])
def test_shell_passes_selected_tasks_and_options_to_finetune(tmp_path, quoted):
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    tasks = [TASK, "robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env"]
    command = [
        "bash",
        "examples/finetune.sh",
        "--base-model-path",
        "base",
        "--dataset-path",
        "data",
        "--embodiment-tag",
        "ROBOCASA_PANDA_OMRON",
        "--output-dir",
        str(tmp_path / "out"),
        "--eval-steps",
        "12",
        "--eval-tasks",
        *([" ".join(tasks)] if quoted else tasks),
        "--eval-python",
        "/sim/python",
        "--eval-n-episodes",
        "7",
        "--eval-seed",
        "4",
        "--",
        "--max-steps",
        "24",
    ]
    env = {**os.environ, "PATH": f"{tmp_path}:{os.defpath}", "NUM_GPUS": "1", "USE_WANDB": "0"}
    result = subprocess.run(
        command, cwd=rollout_eval.REPO_ROOT, env=env, capture_output=True, text=True, check=True
    )
    arguments = json.loads(result.stdout)
    config = tyro.cli(FinetuneConfig, args=arguments[1:])
    assert config.eval_steps == 12
    assert config.eval_tasks == tasks
    assert config.eval_python == "/sim/python"
    assert config.eval_n_episodes == 7
    assert config.eval_seed == 4
    assert config.max_steps == 24


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_modes_rng_and_weights(fail):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model[0].eval()
    modes = [module.training for module in model.modules()]
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    torch_state = torch.random.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with evaluation_state(model, 42):
            assert not any(module.training for module in model.modules())
            assert torch.is_inference_mode_enabled()
            torch.rand(10)
            np.random.rand(10)
            random.random()
            if fail:
                raise RuntimeError("simulator failed")
    except RuntimeError:
        assert fail
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])


def test_policy_wraps_live_weights_without_changing_mode_or_dtype():
    model = torch.nn.Linear(2, 2)
    processor = _processor()
    policy = Gr00tPolicy.from_model("ROBOCASA_PANDA_OMRON", model, processor)
    assert policy.model is model
    assert policy.processor is processor
    assert model.training
    assert model.weight.dtype == torch.float32
    processor.eval.assert_not_called()
    assert policy.language_key == "task"


@pytest.mark.parametrize("step", [0, 1, 4, 6, 11])
def test_disabled_or_off_schedule_does_not_evaluate(tmp_path, step):
    for interval in (0, 5):
        trainer = MagicMock()
        callback = RolloutEvalCallback(
            trainer, RolloutEvalConfig(steps=interval, tasks=[TASK]), tmp_path
        )
        callback._evaluate = MagicMock()
        callback.on_step_end(None, SimpleNamespace(global_step=step), None)
        callback._evaluate.assert_not_called()
        trainer.log.assert_not_called()


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return {"loss": self.layer(x).square().mean()}


def test_real_trainer_logs_live_rollouts_to_same_wandb_run(tmp_path, monkeypatch):
    """Exercise optimizer-step scheduling, checkpoint independence, and HF's W&B bridge."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=3,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to="none",
        save_steps=3,
        logging_steps=1,
        disable_tqdm=True,
    )
    trainer = Gr00tTrainer(
        model=_TinyModel(), args=args, train_dataset=[{"x": torch.ones(1)} for _ in range(12)]
    )
    wandb_callback = WandbCallback()
    wandb_callback._wandb = MagicMock()
    wandb_callback._initialized = True
    trainer.add_callback(wandb_callback)
    processor = _processor()
    monkeypatch.setattr(rollout_eval.AutoProcessor, "from_pretrained", lambda path: processor)
    evaluated = []

    def fake_task(config, task, policy, output_dir):
        evaluated.append((trainer.state.global_step, task))
        assert policy.policy.model is trainer.model
        assert not trainer.model.training
        assert not (tmp_path / "checkpoint-2").exists()
        assert policy.policy.processor is processor
        return _report(task)

    monkeypatch.setattr(rollout_eval, "run_task", fake_task)
    tasks = [TASK, "robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env"]
    trainer.add_callback(
        RolloutEvalCallback(trainer, RolloutEvalConfig(steps=2, tasks=tasks), tmp_path)
    )
    trainer.train()

    assert evaluated == [(2, task) for task in tasks]
    assert (tmp_path / "checkpoint-3").is_dir()
    (metrics_file,) = (tmp_path / "rollout_eval").glob("step-2-*/metrics.json")
    assert json.loads(metrics_file.read_text())["step"] == 2
    logs = [call.args[0] for call in wandb_callback._wandb.log.call_args_list]
    (logged,) = [log for log in logs if "eval/rollout/success_rate" in log]
    assert logged["train/global_step"] == 2
    assert logged[f"eval/rollout/{TASK}/success_rate"] == 0.5
    assert logged[f"eval/rollout/{TASK}/mean_reward"] == 2
    assert logged[f"eval/rollout/{TASK}/mean_length"] == 6
    assert wandb_callback._wandb.init.call_count == 0


@pytest.mark.parametrize("outcome", ["ok", "fail", "timeout"])
def test_rollout_subprocess_transport_and_cleanup(tmp_path, monkeypatch, outcome):
    """Use a real child and ZMQ socket without requiring MuJoCo or a GPU."""
    child_script = tmp_path / "simulator.py"
    child_script.write_text("""
import json
import os
from pathlib import Path
import sys
import time
import msgpack
import zmq

outcome = sys.argv[1]
if outcome == "fail":
    raise RuntimeError("simulator failed")
if outcome == "timeout":
    time.sleep(60)
port = sys.argv[sys.argv.index("--policy-client-port") + 1]
output = Path(sys.argv[sys.argv.index("--video-dir") + 1])
task = sys.argv[sys.argv.index("--env-name") + 1]
assert "RANK" not in os.environ
assert "LD_LIBRARY_PATH" not in os.environ
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:" + port)
socket.send(msgpack.packb({"endpoint": "get_action", "data": {"observation": {}}}))
assert msgpack.unpackb(socket.recv()) == [{"value": 42}, {}]
socket.close()
context.term()
(output / "results_fake.json").write_text(json.dumps({
    "env_name": task, "n_episodes": 1, "n_successes": 1,
    "episode_infos": {"episode_rewards": [1], "episode_lengths": [1]}
}))
""")
    real_popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        process = real_popen([sys.executable, str(child_script), outcome, *command[3:]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(rollout_eval.subprocess, "Popen", launch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/invalid/training/libs")
    policy = MagicMock()
    policy.get_action.return_value = ({"value": 42}, {})
    config = RolloutEvalConfig(steps=5, tasks=[TASK], timeout=1 if outcome == "timeout" else 30)
    if outcome == "ok":
        report = rollout_eval.run_task(config, TASK, policy, tmp_path / "output")
        assert report["n_successes"] == 1
        policy.get_action.assert_called_once_with(observation={})
    else:
        with pytest.raises((RuntimeError, TimeoutError), match="rollout.log"):
            rollout_eval.run_task(config, TASK, policy, tmp_path / "output")
    assert processes[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(processes[0].pid, 0)


def _distributed_worker(rank, rendezvous, output_dir, fail):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        trainer = MagicMock()
        callback = RolloutEvalCallback(
            trainer, RolloutEvalConfig(steps=5, tasks=[TASK]), Path(output_dir)
        )

        def evaluate(args, step):
            assert rank == 0
            if fail:
                raise ValueError("broken simulator")
            return {"eval_rollout/success_rate": 0.5}

        callback._evaluate = evaluate
        state = SimpleNamespace(global_step=10, is_world_process_zero=rank == 0)
        try:
            callback.on_step_end(None, state, None)
            result = trainer.log.call_args.args[0]
        except RuntimeError as exc:
            result = str(exc)
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.serial
@pytest.mark.parametrize("fail", [False, True])
def test_distributed_ranks_receive_metrics_or_error(tmp_path, fail):
    mp.spawn(
        _distributed_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), fail),
        nprocs=2,
        join=True,
    )
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert results[0] == results[1]
    if fail:
        assert "broken simulator" in results[0]
    else:
        assert results[0]["eval_rollout/success_rate"] == 0.5
