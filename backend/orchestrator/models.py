from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .state import SystemState


@dataclass(slots=True)
class SystemConfig:
    target_diameter: float
    pixel_to_micron: float
    video_source_type: str
    video_source: str
    initial_q1: float
    initial_q2: float
    control_interval_ms: int


@dataclass(slots=True)
class RecognitionSnapshot:
    avg_diameter: float
    droplet_count: int
    single_cell_rate: float
    valid_for_control: bool
    timestamp: float


@dataclass(slots=True)
class PumpRuntimeState:
    connected: bool
    comm_established: bool
    fully_ready: bool
    q1: float
    q2: float
    running: bool
    last_error: str


@dataclass(slots=True)
class ControlSnapshot:
    diameter_error: float
    adjustment: float
    q1_command: float
    q2_command: float
    freeze_feedback: bool
    suggested_stop: bool
    reason: str
    timestamp: float


@dataclass(slots=True)
class SystemSnapshot:
    system_state: SystemState
    config: Optional[SystemConfig]
    recognition: Optional[RecognitionSnapshot]
    pump_state: Optional[PumpRuntimeState]
    control: Optional[ControlSnapshot]
    message: str
    error: str

