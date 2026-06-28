from __future__ import annotations

import base64
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime for local video only
    cv2 = None

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
        except TypeError:
            self._call(["prepare_video", "prepare", "setup"], video_source_type, video_source, pixel_to_micron)

    def start(self) -> None:
        self._call(["start", "start_loop", "run"])

    def stop(self) -> None:
        self._call(["stop", "stop_loop", "shutdown"])

    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]:
        return self._call(["get_snapshot", "get_latest_snapshot", "read_snapshot", "pull_result", "run_once"])


class PipelineVisionService:
    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        from ..vision.service import VisionCameraService

        self._log = logger or (lambda _msg: None)
        self._pipeline = None
        self._camera_service = VisionCameraService(logger=self._log)
        self._video_source_type = "camera"
        self._video_source = "0"
        self._selected_backend = ""
        self._pixel_to_micron = 1.0
        self._lock = threading.RLock()
        self._cap = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_processed_frame_id = 0
        self._latest = self._empty_snapshot("当前无有效液滴通过")

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from ..vision.config import default_config
            from ..vision.pipeline import VisionPipeline

            self._pipeline = VisionPipeline(default_config(), logger=self._log)
        return self._pipeline

    def _is_realtime_mode(self) -> bool:
        mode = self._video_source_type.strip().lower()
        return mode in {
            "camera",
            "realtime",
            "real_time",
            "live",
            "usb",
            "opencv",
            "hikrobot",
            "hikrobot_industrial_camera",
            "industrial_camera",
            "usb_camera",
        }

    def _empty_snapshot(self, reason: str) -> RecognitionSnapshot:
        return RecognitionSnapshot(
            frame_droplet_count=0,
            total_droplet_count=0,
            new_crossing_count=0,
            avg_diameter=None,
            single_cell_rate=0.0,
            valid_for_control=False,
            timestamp=time.time(),
            reason=reason,
            droplet_count=0,
            active_droplet_count=0,
            has_droplet=False,
            control_reason=reason,
            frame_png_base64=None,
            frame_width=0,
            frame_height=0,
            video_source_type=self._video_source_type,
            video_source=self._video_source,
        )

    def set_mvs_sdk_path(self, sdk_path: str) -> None:
        self._camera_service.set_mvs_sdk_path(sdk_path)

    def set_selected_backend(self, backend_name: str) -> None:
        self._selected_backend = str(backend_name or "").strip()

    def discover_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.discover_cameras_result()

    def refresh_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.refresh_cameras_result()

    def get_camera_devices(self) -> list[dict[str, Any]]:
        return self._camera_service.get_camera_devices()

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        self._video_source = str(unique_id or "")
        self._selected_backend = str(backend_name or "")
        return self._camera_service.select_camera(unique_id, backend_name)

    def test_camera(self) -> dict[str, Any]:
        return self._camera_service.test_camera()

    def get_camera_status(self) -> dict[str, Any]:
        return self._camera_service.get_camera_status()

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        self.stop()
        with self._lock:
            self._video_source_type = str(video_source_type or "camera")
            self._video_source = str(video_source or "0")
            self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0 else 1.0
            self._last_processed_frame_id = 0
            self._ensure_pipeline().reset()
            self._latest = self._empty_snapshot("视频输入已准备，等待识别")

        if self._is_realtime_mode():
            backend = self._selected_backend or _backend_from_mode(self._video_source_type)
            self._camera_service.select_camera(self._video_source, backend or None)
            self._camera_service.open_camera()
            self._camera_service.start_camera_stream()
            deadline = time.time() + 3.0
            packet = self._camera_service.get_latest_frame()
            while time.time() < deadline and (not packet.valid or packet.image is None):
                time.sleep(0.03)
                packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                raise RuntimeError(packet.error or "实时相机未产生有效帧")

    def _open_capture(self):
        if cv2 is None:
            raise RuntimeError("OpenCV/cv2 未安装，无法读取本地视频")
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self._video_source}")
        return cap

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if not self._is_realtime_mode():
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
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()
        try:
            self._camera_service.stop_camera_stream()
            self._camera_service.close_camera()
        except Exception:
            pass

    def _encode_png_base64(self, frame) -> tuple[str | None, int, int]:
        if cv2 is None:
            return None, 0, 0
        try:
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                return None, int(frame.shape[1]), int(frame.shape[0])
            return base64.b64encode(buf.tobytes()).decode("ascii"), int(frame.shape[1]), int(frame.shape[0])
        except Exception:
            return None, 0, 0

    def _read_next_frame(self) -> tuple[bool, Any, str]:
        if self._is_realtime_mode():
            packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                return False, None, packet.error or "相机取帧异常"
            if packet.frame_id == self._last_processed_frame_id:
                return False, None, ""
            self._last_processed_frame_id = int(packet.frame_id)
            return True, packet.image, ""

        with self._lock:
            cap = self._cap
        if cap is None:
            return False, None, "视频源未打开"
        ok, frame = cap.read()
        if not ok:
            return False, None, "本地视频读取结束"
        return True, frame, ""

    def _snapshot_from_frame(self, frame) -> RecognitionSnapshot:
        result = self._ensure_pipeline().process_frame(frame)
        avg_px = result.metrics.control.average_diameter
        active_count = int(result.metrics.control.frame_droplet_count)
        total_count = int(result.metrics.control.total_droplet_count)
        new_cross = int(result.metrics.control.new_crossing_count)
        has_droplet = active_count > 0
        control_reason = str(result.metrics.control.reason or "")
        frame_b64, width, height = self._encode_png_base64(result.annotated_frame)
        return RecognitionSnapshot(
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

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame, error = self._read_next_frame()
            if not ok:
                if error and self._is_realtime_mode():
                    with self._lock:
                        self._latest = self._empty_snapshot(error)
                elif error:
                    break
                time.sleep(0.03)
                continue
            try:
                snapshot = self._snapshot_from_frame(frame)
                with self._lock:
                    self._latest = snapshot
            except Exception as exc:
                self._log(f"[VISION][WARN] 帧处理失败: {exc}")
                time.sleep(0.02)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()

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


def _backend_from_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    aliases = {"alliedvision": "allied_vision"}
    value = aliases.get(value, value)
    return value if value in {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl", "opencv"} else ""
