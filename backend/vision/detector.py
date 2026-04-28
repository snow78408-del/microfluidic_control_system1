from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .config import DebugConfig, DetectorConfig
except ImportError:
    from config import DebugConfig, DetectorConfig


@dataclass
class DetectionResult:
    centers: List[np.ndarray]
    radii: List[float]
    debug_image: np.ndarray
    helper_mask: np.ndarray


class DropletDetector:
    def __init__(self, config: DetectorConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug

    def detect(self, gray_frame: np.ndarray, mode: Optional[str] = None) -> DetectionResult:
        gray = self._ensure_gray(gray_frame)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        blur_size = self._odd(self._config.gaussian_blur_size)
        smoothed = cv2.GaussianBlur(normalized, (blur_size, blur_size), 0) if blur_size > 1 else normalized

        _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = self._morphology_clean(binary)

        detect_mode = mode or self._config.detection_mode
        cut_line = int(binary.shape[0] * self._config.cut_line_ratio)

        if detect_mode == "no_split":
            centers, radii = self._detect_no_split(binary, cut_line)
        else:
            centers, radii = self._detect_split_connected(binary, cut_line)

        centers, radii = self._deduplicate(centers, radii)
        helper_mask = self._build_bead_helper_mask(normalized)
        debug_image = self._make_debug_image(normalized, binary, centers, radii)

        return DetectionResult(centers=centers, radii=radii, debug_image=debug_image, helper_mask=helper_mask)

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _morphology_clean(self, binary: np.ndarray) -> np.ndarray:
        open_k = self._odd(self._config.morphology_open_kernel)
        close_k = self._odd(self._config.morphology_close_kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        return cleaned

    def _detect_no_split(self, binary: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._config.min_contour_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self._config.circularity_threshold:
                continue

            radius = float(np.sqrt(area / np.pi))
            if radius < self._config.min_radius or radius > self._config.max_radius:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) <= 1e-6:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            if cy > cut_line:
                continue

            centers.append(np.array([cx, cy], dtype=np.float32))
            radii.append(radius)

        return centers, radii

    def _detect_split_connected(self, binary: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8, cv2.CV_32S)

        single_max_area = np.pi * (self._config.max_radius ** 2) * self._config.split_large_area_ratio
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            component_mask = (labels == label).astype(np.uint8) * 255

            if area <= single_max_area:
                c_centers, c_radii = self._detect_no_split(component_mask, cut_line)
            else:
                c_centers, c_radii = self._split_component(component_mask, cut_line)

            centers.extend(c_centers)
            radii.extend(c_radii)

        return centers, radii

    def _split_component(self, component_mask: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        kernel_size = self._odd(max(3, int(self._config.min_radius * 1.2)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(dist, kernel)

        peak_threshold = max(1.0, self._config.min_radius * self._config.split_peak_threshold_ratio)
        local_peaks = ((dist >= (dilated - 1e-6)) & (dist > peak_threshold)).astype(np.uint8)

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_peaks, 8, cv2.CV_32S)
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for label in range(1, num_labels):
            cx, cy = centroids[label]
            x = int(round(cx))
            y = int(round(cy))
            if y < 0 or y >= dist.shape[0] or x < 0 or x >= dist.shape[1]:
                continue

            radius = float(dist[y, x])
            if radius < (self._config.min_radius * 0.6) or radius > (self._config.max_radius * 1.3):
                continue
            if y > cut_line:
                continue

            centers.append(np.array([cx, cy], dtype=np.float32))
            radii.append(radius)

        if centers:
            return centers, radii
        return self._detect_no_split(component_mask, cut_line)

    def _deduplicate(
        self,
        centers: List[np.ndarray],
        radii: List[float],
    ) -> Tuple[List[np.ndarray], List[float]]:
        kept_centers: List[np.ndarray] = []
        kept_radii: List[float] = []

        for center, radius in zip(centers, radii):
            duplicate = False
            for existing in kept_centers:
                if float(np.linalg.norm(center - existing)) < self._config.min_center_distance:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept_centers.append(center)
            kept_radii.append(float(radius))

        return kept_centers, kept_radii

    def _build_bead_helper_mask(self, normalized_gray: np.ndarray) -> np.ndarray:
        threshold_value = float(np.percentile(normalized_gray, 15.0))
        _, helper = cv2.threshold(normalized_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        helper = cv2.morphologyEx(helper, cv2.MORPH_OPEN, kernel)
        return helper

    def _make_debug_image(
        self,
        normalized_gray: np.ndarray,
        binary_mask: np.ndarray,
        centers: List[np.ndarray],
        radii: List[float],
    ) -> np.ndarray:
        debug_image = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
        for center, radius in zip(centers, radii):
            cv2.circle(debug_image, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 2)
            cv2.circle(debug_image, (int(center[0]), int(center[1])), 2, (0, 255, 255), -1)

        if self._debug.draw_helper_mask:
            mask_edges = cv2.Canny(binary_mask, 50, 150)
            debug_image[mask_edges > 0] = (0, 200, 0)

        return debug_image

    def _odd(self, value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
