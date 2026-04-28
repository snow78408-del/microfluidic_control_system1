from dataclasses import dataclass, field
from typing import Literal, Tuple

TrackerType = Literal["nearest", "kalman"]
DetectionMode = Literal["split_connected", "no_split"]
BeadMode = Literal["intensity", "connected"]


@dataclass
class ROIConfig:
    enabled: bool = False
    x_start_ratio: float = 0.0
    x_end_ratio: float = 1.0
    y_start_ratio: float = 0.0
    y_end_ratio: float = 1.0
    crop_top_ratio: float = 0.0

    def resolve(self, width: int, height: int) -> Tuple[int, int, int, int, int]:
        x0 = max(0, min(width - 1, int(width * self.x_start_ratio)))
        x1 = max(x0 + 1, min(width, int(width * self.x_end_ratio)))
        y0 = max(0, min(height - 1, int(height * self.y_start_ratio)))
        y1 = max(y0 + 1, min(height, int(height * self.y_end_ratio)))
        crop_top = max(0, min(y1 - y0 - 1, int((y1 - y0) * self.crop_top_ratio)))
        return x0, x1, y0, y1, crop_top


@dataclass
class DetectorConfig:
    min_radius: float = 18.0
    max_radius: float = 32.0
    min_center_distance: float = 35.0
    circularity_threshold: float = 0.15
    min_contour_area: float = 120.0
    gaussian_blur_size: int = 5
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 5
    split_peak_threshold_ratio: float = 0.45
    split_large_area_ratio: float = 1.35
    cut_line_ratio: float = 1.0
    detection_mode: DetectionMode = "split_connected"


@dataclass
class KalmanConfig:
    process_noise: float = 8.0
    measurement_noise: float = 12.0
    initial_covariance: float = 25.0
    dt: float = 1.0


@dataclass
class TrackerConfig:
    tracker_type: TrackerType = "nearest"
    match_distance: float = 90.0
    max_unmatched_frames: int = 8
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class BeadConfig:
    mode: BeadMode = "intensity"
    area_min: int = 5
    area_max: int = 80
    inner_radius_ratio: float = 0.82
    border_margin: int = 2
    default_droplet_radius: float = 24.0
    dark_percentile: float = 18.0
    blur_kernel: int = 5


@dataclass
class MetricsConfig:
    min_active_for_control: int = 2
    min_samples_for_control: int = 12
    rolling_window: int = 120


@dataclass
class DebugConfig:
    enabled: bool = False
    verbose: bool = False
    draw_helper_mask: bool = True


@dataclass
class PipelineConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    beads: BeadConfig = field(default_factory=BeadConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def default_config() -> PipelineConfig:
    return PipelineConfig()
