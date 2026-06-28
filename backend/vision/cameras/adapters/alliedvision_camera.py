from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class AlliedVisionCameraAdapter(BaseCameraAdapter):
    backend_name = "alliedvision"
    supported_manufacturers = ("Allied Vision", "AlliedVision")
    backend_priority = 14

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._vmb = None
        self._camera = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            _import_vimba(prefer_vmbpy=True)
        except Exception as exc:
            return False, f"Vimba X Python API(vmbpy) 不可用: {exc}"
        return True, "Vimba X 可用"

    @classmethod
    def get_backend_diagnostics(cls, config: Any | None = None) -> dict[str, Any]:
        try:
            module = _import_vimba(prefer_vmbpy=True)
            return {
                "display_name": "Allied Vision",
                "sdk_loaded": True,
                "python_module_loaded": True,
                "python_module_path": str(getattr(module, "__file__", "") or "vmbpy"),
                "error": "",
            }
        except Exception as exc:
            return {
                "display_name": "Allied Vision",
                "sdk_loaded": False,
                "python_module_loaded": False,
                "error": f"Vimba X Python API(vmbpy) 不可用: {exc}",
            }

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        vimba = _import_vimba()
        devices: list[CameraDeviceInfo] = []
        with vimba.VmbSystem.get_instance() as vmb:
            for cam in vmb.get_all_cameras():
                camera_id = cam.get_id()
                name = cam.get_name() if hasattr(cam, "get_name") else camera_id
                serial = camera_id
                model = str(name or "")
                manufacturer = "Allied Vision"
                unique = f"{manufacturer}:{model}:{serial}"
                devices.append(
                    CameraDeviceInfo(
                        device_id=f"alliedvision:{camera_id}",
                        unique_id=unique,
                        backend_name=cls.backend_name,
                        manufacturer=manufacturer,
                        model=model,
                        serial_number=serial,
                        device_type=DEVICE_TYPE_INDUSTRIAL,
                        transport_type="Unknown",
                        available=True,
                        capabilities=_caps(),
                        available_backends=[cls.backend_name],
                        selected_backend=cls.backend_name,
                        backend_priority=cls.backend_priority,
                    )
                )
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        vimba = _import_vimba()
        self.close()
        self._vmb = vimba.VmbSystem.get_instance()
        self._vmb.__enter__()
        camera_id = device_info.device_id.split(":", 1)[1]
        self._camera = self._vmb.get_camera_by_id(camera_id)
        self._camera.__enter__()
        self._device = device_info

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.__exit__(None, None, None)
            except Exception:
                pass
        if self._vmb is not None:
            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass
        self._camera = None
        self._vmb = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=alliedvision")

    def start_stream(self) -> None:
        if self._camera is None:
            raise CameraBackendError("Allied Vision 相机未打开")
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=alliedvision")

    def stop_stream(self) -> None:
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._camera is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="Allied Vision 未开始采集")
        try:
            frame = self._camera.get_frame(timeout_ms=timeout_ms)
            arr = frame.as_numpy_ndarray()
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()
                fmt = "BGR8"
            else:
                fmt = "Mono8"
            self._frame_id += 1
            self._latest = FrameData(arr, self._frame_id, time.time(), int(arr.shape[1]), int(arr.shape[0]), fmt, self.backend_name, self._device.unique_id if self._device else "", True)
            return self._latest
        except Exception as exc:
            self._last_error = str(exc)
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=str(exc))

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return _caps()

    def get_feature(self, name: str) -> Any:
        feature = getattr(self._camera, _node(name), None) if self._camera is not None else None
        return feature.get() if feature is not None and hasattr(feature, "get") else None

    def set_feature(self, name: str, value: Any) -> None:
        feature = getattr(self._camera, _node(name), None) if self._camera is not None else None
        if feature is not None and hasattr(feature, "set"):
            feature.set(value)

    def is_open(self) -> bool:
        return self._camera is not None

    def is_streaming(self) -> bool:
        return self._streaming


def _import_vimba(prefer_vmbpy: bool = False):
    try:
        import vmbpy
        return vmbpy
    except Exception as vmbpy_exc:
        try:
            from vimba import Vimba as _Vimba
        except Exception:
            raise vmbpy_exc
        if prefer_vmbpy:
            # Vimba X should expose vmbpy. Old vimba is accepted only as a compatibility fallback.
            pass

        class _CompatSystem:
            @staticmethod
            def get_instance():
                return _Vimba.get_instance()

        class _Compat:
            VmbSystem = _CompatSystem

        return _Compat


def _node(name: str) -> str:
    return {
        "exposure": "ExposureTime",
        "gain": "Gain",
        "frame_rate": "AcquisitionFrameRate",
        "width": "Width",
        "height": "Height",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "pixel_format": "PixelFormat",
        "trigger_mode": "TriggerMode",
    }.get(name, name)


def _caps() -> CameraCapabilities:
    return CameraCapabilities(
        exposure=CameraFeatureCapability(True, True, True),
        gain=CameraFeatureCapability(True, True, True),
        frame_rate=CameraFeatureCapability(True, True, True),
        width=CameraFeatureCapability(True, True, True),
        height=CameraFeatureCapability(True, True, True),
        offset_x=CameraFeatureCapability(True, True, True),
        offset_y=CameraFeatureCapability(True, True, True),
        pixel_format=CameraFeatureCapability(True, True, True),
        trigger_mode=CameraFeatureCapability(True, True, True),
        acquisition_mode=CameraFeatureCapability(True, True, True, "Continuous"),
    )