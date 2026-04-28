from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np

try:
    from .bead_counter import BeadResult
    from .config import MetricsConfig
    from .tracker import TrackingResult
except ImportError:
    from bead_counter import BeadResult
    from config import MetricsConfig
    from tracker import TrackingResult


@dataclass
class ControlMetrics:
    average_diameter: float
    current_active_droplets: int
    sample_size: int
    valid_for_control: bool
    reason: str


@dataclass
class AnalysisMetrics:
    total_droplets: int
    average_diameter: float
    current_valid_droplets: int
    single_bead_count: int
    single_bead_rate: float
    empty_count: int
    empty_rate: float
    multi_bead_count: int
    multi_bead_rate: float


@dataclass
class MetricsResult:
    control: ControlMetrics
    analysis: AnalysisMetrics


class MetricsCalculator:
    def __init__(self, config: MetricsConfig) -> None:
        self._config = config
        self._diameter_history: Deque[float] = deque(maxlen=max(1, config.rolling_window))
        self._track_bead_max: Dict[int, int] = {}

    def update(self, tracking: TrackingResult, beads: BeadResult) -> MetricsResult:
        for track in tracking.active_tracks:
            if track.radius > 0:
                self._diameter_history.append(float(track.radius * 2.0))

        for droplet in beads.droplets:
            prev_max = self._track_bead_max.get(droplet.droplet_id, 0)
            if droplet.bead_count > prev_max:
                self._track_bead_max[droplet.droplet_id] = droplet.bead_count

        total_droplets = tracking.total_count
        bead_counts = [self._track_bead_max.get(track_id, 0) for track_id in range(1, total_droplets + 1)]

        empty_count = sum(1 for value in bead_counts if value == 0)
        single_count = sum(1 for value in bead_counts if value == 1)
        multi_count = sum(1 for value in bead_counts if value >= 2)

        average_diameter = float(np.mean(self._diameter_history)) if self._diameter_history else 0.0
        current_active = len(tracking.active_tracks)

        sample_size = len(self._diameter_history)
        valid_for_control = True
        reason = "ok"

        if sample_size < self._config.min_samples_for_control:
            valid_for_control = False
            reason = "not_enough_samples"
        elif current_active < self._config.min_active_for_control:
            valid_for_control = False
            reason = "not_enough_active_droplets"

        denom = float(total_droplets) if total_droplets > 0 else 1.0
        analysis = AnalysisMetrics(
            total_droplets=total_droplets,
            average_diameter=average_diameter,
            current_valid_droplets=current_active,
            single_bead_count=single_count,
            single_bead_rate=(single_count / denom) * 100.0 if total_droplets else 0.0,
            empty_count=empty_count,
            empty_rate=(empty_count / denom) * 100.0 if total_droplets else 0.0,
            multi_bead_count=multi_count,
            multi_bead_rate=(multi_count / denom) * 100.0 if total_droplets else 0.0,
        )

        control = ControlMetrics(
            average_diameter=average_diameter,
            current_active_droplets=current_active,
            sample_size=sample_size,
            valid_for_control=valid_for_control,
            reason=reason,
        )

        return MetricsResult(control=control, analysis=analysis)

    def reset(self) -> None:
        self._diameter_history.clear()
        self._track_bead_max.clear()
