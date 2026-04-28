from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .bead_counter import BeadCounter, BeadResult
    from .config import PipelineConfig
    from .detector import DetectionResult, DropletDetector
    from .kalman_tracker import KalmanTracker
    from .metrics import MetricsCalculator, MetricsResult
    from .nearest_tracker import NearestTracker
    from .tracker import BaseTracker, TrackingResult
except ImportError:
    from bead_counter import BeadCounter, BeadResult
    from config import PipelineConfig
    from detector import DetectionResult, DropletDetector
    from kalman_tracker import KalmanTracker
    from metrics import MetricsCalculator, MetricsResult
    from nearest_tracker import NearestTracker
    from tracker import BaseTracker, TrackingResult


@dataclass
class VisionResult:
    frame_index: int
    timestamp: float
    detections: DetectionResult
    tracking: TrackingResult
    beads: BeadResult
    metrics: MetricsResult
    annotated_frame: np.ndarray


class VisionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = DropletDetector(config.detector, config.debug)
        self.bead_counter = BeadCounter(config.beads, config.debug)
        self.metrics = MetricsCalculator(config.metrics)
        self.tracker: BaseTracker = self._build_tracker(config)
        self._frame_index = 0

    def _build_tracker(self, config: PipelineConfig) -> BaseTracker:
        if config.tracker.tracker_type == "kalman":
            return KalmanTracker(config.tracker)
        return NearestTracker(config.tracker)

    def reset(self) -> None:
        self.tracker.reset()
        self.metrics.reset()
        self._frame_index = 0

    def process_frame(self, frame: np.ndarray) -> VisionResult:
        self._frame_index += 1
        roi_frame, _ = self._apply_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY) if roi_frame.ndim == 3 else roi_frame
        detections = self.detector.detect(gray)
        tracking = self.tracker.update(detections.centers, detections.radii)
        beads = self.bead_counter.count(tracking.active_tracks, gray, detections.helper_mask)
        metrics = self.metrics.update(tracking, beads)

        annotated = self._draw_overlay(roi_frame, tracking, beads, metrics)

        return VisionResult(
            frame_index=self._frame_index,
            timestamp=time(),
            detections=detections,
            tracking=tracking,
            beads=beads,
            metrics=metrics,
            annotated_frame=annotated,
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> List[VisionResult]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        return self._process_capture(cap, fps=fps, output_path=output_path, display=display, max_frames=max_frames)

    def process_camera(
        self,
        camera_index: int = 0,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> List[VisionResult]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        return self._process_capture(cap, fps=30.0, output_path=output_path, display=display, max_frames=max_frames)

    def _process_capture(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        output_path: Optional[str],
        display: bool,
        max_frames: Optional[int],
    ) -> List[VisionResult]:
        results: List[VisionResult] = []
        writer: Optional[cv2.VideoWriter] = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.process_frame(frame)
                results.append(result)

                if output_path:
                    if writer is None:
                        h, w = result.annotated_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
                    writer.write(result.annotated_frame)

                if display:
                    cv2.imshow("VisionPipeline", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames is not None and len(results) >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        return results

    def _apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        if not self.config.roi.enabled:
            return frame.copy(), (0, 0)

        h, w = frame.shape[:2]
        x0, x1, y0, y1, crop_top = self.config.roi.resolve(w, h)
        cropped = frame[y0:y1, x0:x1]
        if crop_top > 0:
            cropped = cropped[crop_top:, :]
        return cropped, (x0, y0 + crop_top)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracking: TrackingResult,
        beads: BeadResult,
        metrics: MetricsResult,
    ) -> np.ndarray:
        canvas = frame.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        bead_map = {item.droplet_id: item for item in beads.droplets}

        for track in tracking.active_tracks:
            center = (int(track.position[0]), int(track.position[1]))
            radius = int(max(2.0, float(track.radius)))
            cv2.circle(canvas, center, radius, (255, 0, 0), 2)
            cv2.putText(
                canvas,
                f"ID{track.id}",
                (center[0] + 8, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

            bead_count = bead_map.get(track.id).bead_count if track.id in bead_map else 0
            cv2.putText(
                canvas,
                f"beads:{bead_count}",
                (center[0] + 8, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        for droplet in beads.droplets:
            for bead in droplet.bead_positions:
                cv2.circle(canvas, (int(bead[0]), int(bead[1])), 2, (0, 255, 255), -1)

        stats_lines = [
            f"active={metrics.control.current_active_droplets}",
            f"total={metrics.analysis.total_droplets}",
            f"avg_diameter={metrics.control.average_diameter:.2f}",
            f"valid_for_control={metrics.control.valid_for_control}",
        ]
        for idx, line in enumerate(stats_lines):
            cv2.putText(
                canvas,
                line,
                (10, 24 + 22 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return canvas
