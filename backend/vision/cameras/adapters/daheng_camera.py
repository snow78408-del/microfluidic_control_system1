from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class DahengCameraAdapter(BaseCameraAdapter):
    backend_name = "daheng"
    supported_manufacturers = ("Daheng Imaging", "Daheng", "大恒图像")
    backend_priority = 12

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._manager = None
        self._camera = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            import gxipy  # noqa: F401
        except Exception as exc:
            return False, f"Galaxy SDK/gxipy 不可用: {exc}"
        return True, "Galaxy SDK 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        import gxipy as gx

        manager = gx.DeviceManager()
        _, infos = manager.update_device_list()
        devices: list[CameraDeviceInfo] = []
        for info in infos:
            manufacturer = str(info.get("vendor_name", "") or "Daheng Imaging")
            model = str(info.get("model_name", "") or "")
            serial = str(info.get("sn", "") or "")
            transport = str(info.get("device_class", "") or "Unknown")
            unique = f"{manufacturer}:{model}:{serial}" if serial else f"daheng:{info.get('device_id', model)}"
            devices.append(
                CameraDeviceInfo(
                    device_id=f"daheng:{serial or info.get('device_id', '')}",
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=manufacturer,
                    model=model,
                    serial_number=serial,
                    user_defined_name=str(info.get("user_id", "") or ""),
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=_transport(transport),
                    ip_address=str(info.get("ip", "") or ""),
                    available=True,
                    capabilities=_caps(),
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    raw_info=info,
                )
            )
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        import gxipy as gx

        self.close()
        self._manager = gx.DeviceManager()
        self._manager.update_device_list()
        if device_info.serial_number:
            self._camera = self._manager.open_device_by_sn(device_info.serial_number)
        else:
            self._camera = self._manager.open_device_by_index(1)
        self._device = device_info

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.close_device()
            except Exception:
                pass
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=daheng")

    def start_stream(self) -> None:
        if self._camera is None:
            raise CameraBackendError("大恒相机未打开")
        self._camera.stream_on()
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=daheng")

    def stop_stream(self) -> None:
        if self._camera is not None and self._streaming:
            try:
                self._camera.stream_off()
            except Exception:
                pass
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._camera is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒相机未开始采集")
        try:
            raw = self._camera.data_stream[0].get_image(timeout_ms)
            if raw is None:
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒相机取帧超时")
            image = raw.convert("RGB").get_numpy_array()
            if image is None:
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒图像转换失败")
            image = image[:, :, ::-1].copy()
            self._frame_id += 1
            self._latest = FrameData(
                image=image,
                frame_id=self._frame_id,
                timestamp=time.time(),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                pixel_format="BGR8",
                source_backend=self.backend_name,
                device_unique_id=self._device.unique_id if self._device else "",
                valid=True,
            )
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
        return feature.get() if feature is not None else None

    def set_feature(self, name: str, value: Any) -> None:
        feature = getattr(self._camera, _node(name), None) if self._camera is not None else None
        if feature is not None:
            feature.set(value)

    def is_open(self) -> bool:
        return self._camera is not None

    def is_streaming(self) -> bool:
        return self._streaming


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GEV" in v or "GIGE" in v:
        return "GigE"
    return value or "Unknown"


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
        exposure_auto=CameraFeatureCapability(True, True, True),
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

