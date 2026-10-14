"""Evaluate live training weights through the existing simulator RPC interface."""

from contextlib import contextmanager
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoProcessor, TrainerCallback

from gr00t.configs.rollout_eval_config import RolloutEvalConfig
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def evaluation_state(model, seed: int):
    """Restore training RNGs and each module's mode, including frozen submodules."""
    modes = [(module, module.training) for module in model.modules()]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            model.eval()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ),
            ):
                yield
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def simulator_environment() -> dict[str, str]:
    """Keep the simulator independent of torchrun and the training conda runtime."""
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("TORCHELASTIC_", "WANDB_")) or name in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "PYTHONHOME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
        }:
            env.pop(name, None)
    env.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        WANDB_MODE="disabled",
    )
    # Permit an uninstalled checkout while leaving the simulator's site-packages in charge.
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _stop_process_group(process: subprocess.Popen) -> None:
    # The rollout can own async simulator workers; reap them even if its leader exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_task(config: RolloutEvalConfig, task: str, policy, output_dir: Path) -> dict:
    """Serve inference on this thread while a separate interpreter runs the simulator."""
    output_dir.mkdir(parents=True)
    log_path = output_dir / "rollout.log"
    with PolicyServer(policy, host="127.0.0.1", port=0) as server:
        command = [
            config.python or sys.executable,
            "-m",
            "gr00t.eval.rollout_policy",
            "--policy-client-host",
            "127.0.0.1",
            "--policy-client-port",
            str(server.port),
            "--env-name",
            task,
            "--n-episodes",
            str(config.n_episodes),
            "--n-envs",
            str(config.n_envs),
            "--max-episode-steps",
            str(config.max_episode_steps),
            "--n-action-steps",
            str(config.n_action_steps),
            "--seed",
            str(config.seed),
            "--video-dir",
            str(output_dir),
            "--no-video",
        ]
        if config.robocasa_split:
            command.extend(["--robocasa-split", config.robocasa_split])
        logger.info("Rollout evaluation: %s; simulator log: %s", task, log_path)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=simulator_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + config.timeout
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Evaluation timed out for {task}; see {log_path}")
                    server.run_once(timeout_ms=100)
                if process.returncode:
                    raise RuntimeError(
                        f"Evaluation failed for {task} (exit {process.returncode}); see {log_path}"
                    )
            finally:
                _stop_process_group(process)
    reports = list(output_dir.glob("results_*.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one evaluation result for {task}; see {log_path}")
    report = json.loads(reports[0].read_text())
    if report["env_name"] != task or report["n_episodes"] < 1:
        raise ValueError(f"Invalid or empty evaluation results: {reports[0]}")
    return report


def report_metrics(task: str, report: dict) -> dict[str, float]:
    """HF's W&B integration rewrites eval_ to eval/ for the same training run."""
    prefix = f"eval_rollout/{task}"
    count = report["n_episodes"]
    successes = report["n_successes"]
    if count < 1 or not 0 <= successes <= count:
        raise ValueError(f"Invalid episode counts for {task}")
    metrics = {
        f"{prefix}/success_rate": successes / count,
        f"{prefix}/n_successes": successes,
        f"{prefix}/n_episodes": count,
    }
    for field, metric in (("episode_rewards", "mean_reward"), ("episode_lengths", "mean_length")):
        values = report["episode_infos"][field]
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} in evaluation results for {task}")
        metrics[f"{prefix}/{metric}"] = float(np.mean(values))
    return metrics


class RolloutEvalCallback(TrainerCallback):
    """Pause training at optimizer-step boundaries and log selected task rollouts."""

    def __init__(self, trainer, config: RolloutEvalConfig, processor_dir: Path):
        config.validate()
        self.trainer = trainer
        self.config = config
        self.processor_dir = processor_dir
        self.processor = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.config.steps:
            return
        executable = self.config.python or sys.executable
        if shutil.which(executable) is None:
            raise ValueError(f"Simulator Python not found: {executable}; set --eval-python")
        for task in self.config.tasks:
            get_embodiment_tag_from_env_name(task)
        if self.trainer.is_fsdp_enabled or (
            self.trainer.is_deepspeed_enabled
            and self.trainer.model_wrapped.zero_optimization_stage() == 3
        ):
            raise ValueError("Rollout evaluation supports single GPU, DDP, and DeepSpeed ZeRO 1/2")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.config.steps or not state.global_step or state.global_step % self.config.steps:
            return

        # A rollout can exceed NCCL's collective timeout. Peers wait on a separate
        # CPU group with an explicit budget, leaving no pending NCCL collective.
        group = None
        if dist.is_available() and dist.is_initialized():
            group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=self.config.timeout * len(self.config.tasks) + 300),
            )
        payload = [None, None]
        error = None
        try:
            if state.is_world_process_zero:
                try:
                    payload[0] = self._evaluate(args, state.global_step)
                except Exception as exc:
                    error = exc
                    payload[1] = f"{type(exc).__name__}: {exc}"
            if group is not None:
                dist.broadcast_object_list(payload, src=0, group=group)
        finally:
            if group is not None:
                dist.destroy_process_group(group)
        if payload[1] is not None:
            raise RuntimeError(f"Rollout evaluation failed: {payload[1]}") from error
        # Log on every rank to keep TrainerState.log_history consistent. HF's
        # W&B callback itself only emits from world process zero.
        self.trainer.log(payload[0])

    def _evaluate(self, args, step: int) -> dict[str, float]:
        model = self.trainer.accelerator.unwrap_model(self.trainer.model_wrapped)
        root = Path(args.output_dir).resolve() / "rollout_eval"
        root.mkdir(parents=True, exist_ok=True)
        # Fresh directories avoid stale results after resuming or restarting a run.
        output_dir = Path(tempfile.mkdtemp(prefix=f"step-{step}-", dir=root))
        metrics = {}
        started = time.monotonic()
        with evaluation_state(model, self.config.seed):
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.processor_dir)
                self.processor.eval()
            for task in self.config.tasks:
                tag = get_embodiment_tag_from_env_name(task)
                policy = Gr00tSimPolicyWrapper(Gr00tPolicy.from_model(tag, model, self.processor))
                # Reset policy noise per task as well as per evaluation round.
                with evaluation_state(model, self.config.seed):
                    report = run_task(self.config, task, policy, output_dir / quote(task, safe=""))
                metrics.update(report_metrics(task, report))
        metrics["eval_rollout/success_rate"] = float(
            np.mean([metrics[f"eval_rollout/{task}/success_rate"] for task in self.config.tasks])
        )
        metrics["eval_rollout/runtime"] = time.monotonic() - started
        (output_dir / "metrics.json").write_text(json.dumps({"step": step, **metrics}, indent=2))
        logger.info("Rollout evaluation at step %d: %s", step, metrics)
        return metrics
