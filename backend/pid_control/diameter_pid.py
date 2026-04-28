from __future__ import annotations

import math

from .base import BaseDiameterController
from .config import PIDConfig
from .models import PIDCommand, PumpState, TargetParams, VisionMetrics


class DiameterPIDController(BaseDiameterController):
    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self.kp = float(self.config.kp)
        self.ki = float(self.config.ki)
        self.kd = float(self.config.kd)
        self.integral = 0.0
        self.previous_error = 0.0
        self._has_previous = False

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self._has_previous = False

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _is_invalid_number(value: float) -> bool:
        return (value is None) or (not math.isfinite(float(value)))

    def update(
        self,
        vision_metrics: VisionMetrics,
        target_params: TargetParams,
        pump_state: PumpState,
        dt: float,
    ) -> PIDCommand:
        if dt <= 0:
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=0.0,
                adjustment=0.0,
                freeze_feedback=True,
                suggested_stop=False,
                reason="冻结反馈: dt <= 0",
            )

        if not bool(vision_metrics.valid_for_control):
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=0.0,
                adjustment=0.0,
                freeze_feedback=True,
                suggested_stop=False,
                reason="冻结反馈: vision_metrics.valid_for_control=False",
            )

        if int(vision_metrics.droplet_count) < int(self.config.min_droplet_count_for_feedback):
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=0.0,
                adjustment=0.0,
                freeze_feedback=True,
                suggested_stop=False,
                reason=(
                    "冻结反馈: droplet_count不足 "
                    f"({vision_metrics.droplet_count} < {self.config.min_droplet_count_for_feedback})"
                ),
            )

        avg_diameter = float(vision_metrics.avg_diameter)
        target_diameter = float(target_params.target_diameter)
        if self._is_invalid_number(avg_diameter) or avg_diameter <= 0:
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=0.0,
                adjustment=0.0,
                freeze_feedback=True,
                suggested_stop=False,
                reason="冻结反馈: avg_diameter 无效",
            )
        if self._is_invalid_number(target_diameter) or target_diameter <= 0:
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=0.0,
                adjustment=0.0,
                freeze_feedback=True,
                suggested_stop=False,
                reason="冻结反馈: target_diameter 无效",
            )

        error = target_diameter - avg_diameter
        if abs(error) <= float(self.config.diameter_deadband):
            self.previous_error = error
            self._has_previous = True
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=error,
                adjustment=0.0,
                freeze_feedback=False,
                suggested_stop=False,
                reason="直径误差在死区内，无需调节",
            )

        self.integral += error * dt
        self.integral = self._clamp(
            self.integral,
            -abs(float(self.config.integral_limit)),
            abs(float(self.config.integral_limit)),
        )
        derivative = 0.0 if not self._has_previous else (error - self.previous_error) / dt
        raw_adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        adjustment = self._clamp(
            raw_adjustment,
            float(self.config.adjustment_min),
            float(self.config.adjustment_max),
        )

        q1_raw = float(pump_state.q1) + adjustment
        q2_raw = float(pump_state.q2) + adjustment
        self.previous_error = error
        self._has_previous = True

        if q1_raw <= 0 or q2_raw <= 0:
            return PIDCommand(
                q1=float(pump_state.q1),
                q2=float(pump_state.q2),
                diameter_error=error,
                adjustment=adjustment,
                freeze_feedback=False,
                suggested_stop=True,
                reason=(
                    "计算后出现非法流速(<=0)，建议停机: "
                    f"q1_raw={q1_raw:.6f}, q2_raw={q2_raw:.6f}"
                ),
            )

        q1_new = self._clamp(q1_raw, float(self.config.q1_min), float(self.config.q1_max))
        q2_new = self._clamp(q2_raw, float(self.config.q2_min), float(self.config.q2_max))
        return PIDCommand(
            q1=q1_new,
            q2=q2_new,
            diameter_error=error,
            adjustment=adjustment,
            freeze_feedback=False,
            suggested_stop=False,
            reason="PID 调节完成",
        )

