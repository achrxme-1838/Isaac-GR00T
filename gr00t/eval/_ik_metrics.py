"""Cumulative IK diagnostics, including reset and every inner environment step."""

from dataclasses import dataclass

import numpy as np


IK_METRIC_KEYS = (
    "exp_adap_ik_samples",
    "exp_adap_ik_failures",
    "exp_adap_ik_failure_rate",
    "exp_adap_ik_max_position_error_m",
    "exp_adap_ik_max_rotation_error_rad",
)


@dataclass
class IKDiagnostics:
    samples: int = 0
    failures: int = 0
    max_position_error_m: float = 0.0
    max_rotation_error_rad: float = 0.0

    def record(self, position_error, rotation_error, converged):
        self.samples += 1
        self.failures += int(not converged)
        self.max_position_error_m = max(self.max_position_error_m, position_error)
        self.max_rotation_error_rad = max(self.max_rotation_error_rad, rotation_error)
        return {
            "exp_adap_ik_samples": self.samples,
            "exp_adap_ik_failures": self.failures,
            "exp_adap_ik_failure_rate": self.failures / self.samples,
            "exp_adap_ik_max_position_error_m": self.max_position_error_m,
            "exp_adap_ik_max_rotation_error_rad": self.max_rotation_error_rad,
        }


def extract_ik_metrics(info, env_idx=None):
    """Take the latest cumulative counters after temporal / vector info stacking."""
    if IK_METRIC_KEYS[0] not in info:
        return {}
    metrics = {}
    for key in IK_METRIC_KEYS:
        value = info[key] if env_idx is None else info[key][env_idx]
        metrics[key] = np.asarray(value).reshape(-1)[-1].item()
    return metrics


def format_ik_metrics(metrics):
    return (
        f"IK (reset + steps): samples={metrics['exp_adap_ik_samples']}, "
        f"outside_tolerance={metrics['exp_adap_ik_failures']} "
        f"({metrics['exp_adap_ik_failure_rate']:.2%}), "
        f"max_pos={1000 * metrics['exp_adap_ik_max_position_error_m']:.3f} mm, "
        f"max_rot={np.degrees(metrics['exp_adap_ik_max_rotation_error_rad']):.3f} deg"
    )
