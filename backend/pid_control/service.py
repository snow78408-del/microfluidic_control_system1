from __future__ import annotations

from .base import BaseDiameterController
from .config import PIDConfig
from .diameter_pid import DiameterPIDController
from .models import PIDCommand, PumpState, TargetParams, VisionMetrics

_controller: BaseDiameterController | None = None


def build_controller(config: PIDConfig | None = None) -> BaseDiameterController:
    global _controller
    _controller = DiameterPIDController(config=config)
    return _controller


def reset_controller() -> None:
    global _controller
    if _controller is None:
        _controller = DiameterPIDController()
    _controller.reset()


def run_feedback_step(
    vision_metrics: VisionMetrics,
    target_params: TargetParams,
    pump_state: PumpState,
    dt: float,
) -> PIDCommand:
    global _controller
    if _controller is None:
        _controller = DiameterPIDController()
    return _controller.update(
        vision_metrics=vision_metrics,
        target_params=target_params,
        pump_state=pump_state,
        dt=float(dt),
    )

