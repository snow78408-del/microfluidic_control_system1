from __future__ import annotations

from enum import Enum


class SystemState(str, Enum):
    IDLE = "IDLE"
    CONFIGURED = "CONFIGURED"
    VIDEO_READY = "VIDEO_READY"
    INITIALIZING = "INITIALIZING"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

