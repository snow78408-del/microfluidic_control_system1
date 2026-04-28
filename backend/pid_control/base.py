from __future__ import annotations

from abc import ABC, abstractmethod

from .models import PIDCommand, PumpState, TargetParams, VisionMetrics


class BaseDiameterController(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        vision_metrics: VisionMetrics,
        target_params: TargetParams,
        pump_state: PumpState,
        dt: float,
    ) -> PIDCommand:
        raise NotImplementedError

