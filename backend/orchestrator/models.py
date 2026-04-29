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
    pump_port: str = ""
    pump_address: int = 1
    pump_baudrate: int = 1200
    pump_parity: str = "E"


@dataclass(slots=True)
class RecognitionSnapshot:
    frame_droplet_count: int
    total_droplet_count: int
    new_crossing_count: int
    avg_diameter: float | None
    single_cell_rate: float
    valid_for_control: bool
    timestamp: float
    reason: str
    # backward-compatible mirrors
    droplet_count: int
    active_droplet_count: int
    has_droplet: bool
    control_reason: str
    frame_png_base64: Optional[str] = None
    frame_width: int = 0
    frame_height: int = 0
    video_source_type: str = ""
    video_source: str = ""


@dataclass(slots=True)
class PumpRuntimeState:
    connected: bool
    comm_established: bool
    fully_ready: bool
    q1: float
    q2: float
    running: bool
    last_error: str
    last_update_ok: bool = False
    last_update_reason: str = ""


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
