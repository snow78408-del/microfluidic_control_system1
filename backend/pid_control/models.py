from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VisionMetrics:
    avg_diameter: float
    droplet_count: int
    valid_for_control: bool


@dataclass(slots=True)
class TargetParams:
    target_diameter: float


@dataclass(slots=True)
class PumpState:
    q1: float
    q2: float


@dataclass(slots=True)
class PIDCommand:
    q1: float
    q2: float
    diameter_error: float
    adjustment: float
    freeze_feedback: bool
    suggested_stop: bool
    reason: str

