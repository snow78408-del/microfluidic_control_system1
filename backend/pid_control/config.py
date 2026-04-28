from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PIDConfig:
    kp: float = 0.08
    ki: float = 0.01
    kd: float = 0.0

    q1_min: float = 0.0
    q1_max: float = 5000.0
    q2_min: float = 0.0
    q2_max: float = 5000.0

    adjustment_min: float = -500.0
    adjustment_max: float = 500.0
    diameter_deadband: float = 1.0
    min_droplet_count_for_feedback: int = 5
    integral_limit: float = 10000.0

