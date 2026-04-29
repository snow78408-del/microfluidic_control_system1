from __future__ import annotations

import base64
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

import cv2

from .models import RecognitionSnapshot


@runtime_checkable
class VisionAdapterProtocol(Protocol):
    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]: ...


class GenericVisionAdapter:
    def __init__(self, vision_service: Any) -> None:
        self.vision_service = vision_service

    def _call(self, names: list[str], *args, **kwargs):
        if self.vision_service is None:
            raise RuntimeError("未注入 vision_service")
        for name in names:
            fn = getattr(self.vision_service, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"vision_service 缺少可用接口: {names}")

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        try:
            self._call(
                ["prepare_video", "prepare", "setup"],
                video_source_type=video_source_type,
                video_source=video_source,
                pixel_to_micron=pixel_to_micron,
            )
            return
        except TypeError:
            self._call(["prepare_video", "prepare", "setup"], video_source_type, video_source, pixel_to_micron)

    def start(self) -> None:
        self._call(["start", "start_loop", "run"])

    def stop(self) -> None:
        self._call(["stop", "stop_loop", "shutdown"])

    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]:
        return self._call(["get_snapshot", "get_latest_snapshot", "read_snapshot", "pull_result", "run_once"])


class PipelineVisionService:
    """将 backend/vision 的 VisionPipeline 封装成 orchestrator 可直接调用的服务。"""

    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        from ..vision.config import default_config
        from ..vision.pipeline import VisionPipeline

        self._log = logger or (lambda _msg: None)
        self._pipeline = VisionPipeline(default_config(), logger=self._log)

        self._video_source_type = "camera"
        self._video_source = "0"
        self._pixel_to_micron = 1.0

        self._lock = threading.RLock()
        self._cap: cv2.VideoCapture | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest = RecognitionSnapshot(
            frame_droplet_count=0,
            total_droplet_count=0,
            new_crossing_count=0,
            avg_diameter=None,
            single_cell_rate=0.0,
            valid_for_control=False,
            timestamp=time.time(),
            reason="当前无有效液滴通过",
            droplet_count=0,
            active_droplet_count=0,
            has_droplet=False,
            control_reason="当前无有效液滴通过",
            frame_png_base64=None,
            frame_width=0,
            frame_height=0,
            video_source_type="camera",
            video_source="0",
        )

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        with self._lock:
            self._video_source_type = str(video_source_type or "camera")
            self._video_source = str(video_source or "0")
            self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0 else 1.0
            self._pipeline.reset()

    def _open_capture(self) -> cv2.VideoCapture:
        mode = self._video_source_type.strip().lower()
        if mode in {"camera", "realtime", "real_time", "live", "usb"}:
            source = int(float(self._video_source.strip() or "0"))
        else:
            source = self._video_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self._video_source}")
        return cap

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._cap = self._open_capture()
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._loop, name="vision-pipeline-loop", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = None
            self._worker = None

    def _encode_png_base64(self, frame) -> tuple[str | None, int, int]:
        try:
            ok, buf = cv2.imencode('.png', frame)
            if not ok:
                return None, int(frame.shape[1]), int(frame.shape[0])
            return base64.b64encode(buf.tobytes()).decode('ascii'), int(frame.shape[1]), int(frame.shape[0])
        except Exception:
            return None, 0, 0

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                cap = self._cap
                mode = self._video_source_type.strip().lower()
            if cap is None:
                break

            ok, frame = cap.read()
            if not ok:
                if mode in {"camera", "realtime", "real_time", "live", "usb"}:
                    time.sleep(0.05)
                    continue
                break

            try:
                result = self._pipeline.process_frame(frame)
                avg_px = result.metrics.control.average_diameter
                active_count = int(result.metrics.control.frame_droplet_count)
                total_count = int(result.metrics.control.total_droplet_count)
                new_cross = int(result.metrics.control.new_crossing_count)
                has_droplet = active_count > 0
                control_reason = str(result.metrics.control.reason or "")
                frame_b64, width, height = self._encode_png_base64(result.annotated_frame)
                snapshot = RecognitionSnapshot(
                    frame_droplet_count=active_count,
                    total_droplet_count=total_count,
                    new_crossing_count=new_cross,
                    avg_diameter=(float(avg_px) * self._pixel_to_micron) if avg_px is not None else None,
                    single_cell_rate=float(result.metrics.analysis.single_bead_rate),
                    valid_for_control=bool(result.metrics.control.valid_for_control and has_droplet),
                    timestamp=time.time(),
                    reason=control_reason,
                    droplet_count=total_count,
                    active_droplet_count=active_count,
                    has_droplet=has_droplet,
                    control_reason=control_reason,
                    frame_png_base64=frame_b64,
                    frame_width=width,
                    frame_height=height,
                    video_source_type=self._video_source_type,
                    video_source=self._video_source,
                )
                with self._lock:
                    self._latest = snapshot
            except Exception as e:
                self._log(f"[VISION][WARN] 帧处理失败: {e}")
                time.sleep(0.02)

        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = None
            self._worker = None

    def get_snapshot(self) -> RecognitionSnapshot:
        with self._lock:
            return RecognitionSnapshot(
                frame_droplet_count=self._latest.frame_droplet_count,
                total_droplet_count=self._latest.total_droplet_count,
                new_crossing_count=self._latest.new_crossing_count,
                avg_diameter=self._latest.avg_diameter,
                single_cell_rate=self._latest.single_cell_rate,
                valid_for_control=self._latest.valid_for_control,
                timestamp=self._latest.timestamp,
                reason=self._latest.reason,
                droplet_count=self._latest.droplet_count,
                active_droplet_count=self._latest.active_droplet_count,
                has_droplet=self._latest.has_droplet,
                control_reason=self._latest.control_reason,
                frame_png_base64=self._latest.frame_png_base64,
                frame_width=self._latest.frame_width,
                frame_height=self._latest.frame_height,
                video_source_type=self._latest.video_source_type,
                video_source=self._latest.video_source,
            )

    def run_once(self) -> RecognitionSnapshot:
        return self.get_snapshot()
