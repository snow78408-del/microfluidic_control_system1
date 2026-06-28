from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class FlirCameraAdapter(BaseCameraAdapter):
    backend_name = "flir"
    supported_manufacturers = ("Teledyne FLIR", "FLIR")
    backend_priority = 13

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._system = None
        self._camera_list = None
        self._camera = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            import PySpin  # noqa: F401
        except Exception as exc:
            return False, f"Spinnaker SDK/PySpin 不可用: {exc}"
        return True, "Spinnaker SDK 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        import PySpin

        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        devices: list[CameraDeviceInfo] = []
        try:
            for idx, cam in enumerate(cam_list):
                nodemap = cam.GetTLDeviceNodeMap()
                manufacturer = _node_string(PySpin, nodemap, "DeviceVendorName") or "Teledyne FLIR"
                model = _node_string(PySpin, nodemap, "DeviceModelName")
                serial = _node_string(PySpin, nodemap, "DeviceSerialNumber")
                transport = _node_string(PySpin, nodemap, "DeviceType") or "Unknown"
                unique = f"{manufacturer}:{model}:{serial}" if serial else f"flir:{idx}"
                devices.append(
                    CameraDeviceInfo(
                        device_id=f"flir:{serial or idx}",
                        unique_id=unique,
                        backend_name=cls.backend_name,
                        manufacturer=manufacturer,
                        model=model,
                        serial_number=serial,
                        device_type=DEVICE_TYPE_INDUSTRIAL,
                        transport_type=_transport(transport),
                        available=True,
                        capabilities=_caps(),
                        available_backends=[cls.backend_name],
                        selected_backend=cls.backend_name,
                        backend_priority=cls.backend_priority,
                    )
                )
        finally:
            cam_list.Clear()
            system.ReleaseInstance()
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        import PySpin

        self.close()
        self._system = PySpin.System.GetInstance()
        self._camera_list = self._system.GetCameras()
        target = None
        for idx, cam in enumerate(self._camera_list):
            nodemap = cam.GetTLDeviceNodeMap()
            serial = _node_string(PySpin, nodemap, "DeviceSerialNumber")
            if serial == device_info.serial_number or device_info.device_id.endswith(str(idx)):
                target = cam
                break
        if target is None:
            raise CameraBackendError("未找到 FLIR 相机")
        self._camera = target
        self._camera.Init()
        self._device = device_info

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.DeInit()
            except Exception:
                pass
        if self._camera_list is not None:
            try:
                self._camera_list.Clear()
            except Exception:
                pass
        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass
        self._system = None
        self._camera_list = None
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=flir")

    def start_stream(self) -> None:
        if self._camera is None:
            raise CameraBackendError("FLIR 相机未打开")
        self._camera.BeginAcquisition()
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=flir")

    def stop_stream(self) -> None:
        if self._camera is not None and self._streaming:
            try:
                self._camera.EndAcquisition()
            except Exception:
                pass
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._camera is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="FLIR 未开始采集")
        image = None
        try:
            image = self._camera.GetNextImage(timeout_ms)
            if image.IsIncomplete():
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="FLIR 图像不完整")
            arr = image.GetNDArray()
            if arr.ndim == 2:
                out = arr
                fmt = "Mono8"
            else:
                out = arr[:, :, ::-1].copy()
                fmt = "BGR8"
            self._frame_id += 1
            self._latest = FrameData(out, self._frame_id, time.time(), int(out.shape[1]), int(out.shape[0]), fmt, self.backend_name, self._device.unique_id if self._device else "", True)
            return self._latest
        except Exception as exc:
            self._last_error = str(exc)
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=str(exc))
        finally:
            if image is not None:
                image.Release()

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return _caps()

    def get_feature(self, name: str) -> Any:
        return None

    def set_feature(self, name: str, value: Any) -> None:
        return

    def is_open(self) -> bool:
        return self._camera is not None

    def is_streaming(self) -> bool:
        return self._streaming


def _node_string(PySpin, nodemap, name: str) -> str:
    try:
        node = PySpin.CStringPtr(nodemap.GetNode(name))
        if PySpin.IsReadable(node):
            return str(node.GetValue())
    except Exception:
        return ""
    return ""


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GIGE" in v or "GEV" in v:
        return "GigE"
    return value or "Unknown"


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

