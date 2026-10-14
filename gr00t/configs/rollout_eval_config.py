"""Periodic simulator evaluation, independent of checkpoint saving and validation loss."""

from dataclasses import dataclass, field


@dataclass
class RolloutEvalConfig:
    steps: int = 0
    """Optimizer steps between evaluations; zero disables rollout evaluation."""

    tasks: list[str] = field(default_factory=list)
    """Full Gym environment IDs accepted by gr00t.eval.rollout_policy."""

    python: str | None = None
    """Simulator Python executable. None uses the training interpreter."""

    n_episodes: int = 10
    n_envs: int = 1
    max_episode_steps: int = 720
    n_action_steps: int = 8
    seed: int = 0
    robocasa_split: str = ""
    timeout: int = 3600
    """Maximum wall time per task in seconds, including simulator startup."""

    def validate(self) -> None:
        if self.steps < 0:
            raise ValueError("eval_steps must be >= 0")
        if self.steps == 0:
            return
        if not self.tasks or any(not task.strip() for task in self.tasks):
            raise ValueError("eval_tasks must contain at least one full Gym environment ID")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("eval_tasks must not contain duplicate environment IDs")
        for name in ("n_episodes", "n_envs", "max_episode_steps", "n_action_steps", "timeout"):
            if getattr(self, name) < 1:
                raise ValueError(f"eval_{name} must be >= 1")
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("eval_seed must be in [0, 2**32)")
