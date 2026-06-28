from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class BaslerCameraAdapter(BaseCameraAdapter):
    backend_name = "basler"
    supported_manufacturers = ("Basler",)
    backend_priority = 11

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._camera = None
        self._converter = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            from pypylon import pylon  # noqa: F401
        except Exception as exc:
            return False, f"pypylon/pylon Runtime 不可用: {exc}"
        return True, "pypylon 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        from pypylon import pylon

        devices: list[CameraDeviceInfo] = []
        for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
            manufacturer = dev.GetVendorName() or "Basler"
            model = dev.GetModelName() or ""
            serial = dev.GetSerialNumber() or ""
            transport = dev.GetDeviceClass() or "Unknown"
            unique = f"{manufacturer}:{model}:{serial}" if serial else f"basler:{dev.GetFullName()}"
            devices.append(
                CameraDeviceInfo(
                    device_id=f"basler:{serial or dev.GetFullName()}",
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=manufacturer,
                    model=model,
                    serial_number=serial,
                    user_defined_name=dev.GetUserDefinedName() or "",
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=_transport(transport),
                    available=True,
                    capabilities=_caps(),
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    raw_info=dev,
                )
            )
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        from pypylon import pylon

        self.close()
        factory = pylon.TlFactory.GetInstance()
        target = None
        for dev in factory.EnumerateDevices():
            if device_info.serial_number and dev.GetSerialNumber() == device_info.serial_number:
                target = dev
                break
            if dev.GetFullName() in device_info.device_id:
                target = dev
                break
        if target is None:
            raise CameraBackendError("未找到 Basler 相机")
        self._camera = pylon.InstantCamera(factory.CreateDevice(target))
        self._camera.Open()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._device = device_info

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.Close()
            except Exception:
                pass
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=basler")

    def start_stream(self) -> None:
        from pypylon import pylon

        if self._camera is None:
            raise CameraBackendError("Basler 相机未打开")
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._log("[CAMERA][STREAM][START] backend=basler")

    def stop_stream(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        from pypylon import pylon

        if self._camera is None or not self._camera.IsGrabbing():
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="Basler 未开始采集")
        result = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        try:
            if not result or not result.GrabSucceeded():
                err = result.GetErrorDescription() if result else "Basler 取帧超时"
                self._last_error = err
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=err)
            image = self._converter.Convert(result).GetArray()
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
        finally:
            if result:
                result.Release()

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return _caps()

    def get_feature(self, name: str) -> Any:
        if self._camera is None:
            return None
        node = getattr(self._camera, _node(name), None)
        return node.GetValue() if node is not None else None

    def set_feature(self, name: str, value: Any) -> None:
        if self._camera is None:
            return
        node = getattr(self._camera, _node(name), None)
        if node is not None:
            node.SetValue(value)

    def is_open(self) -> bool:
        return bool(self._camera is not None and self._camera.IsOpen())

    def is_streaming(self) -> bool:
        return bool(self._camera is not None and self._camera.IsGrabbing())


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GIGE" in v or "GEV" in v:
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
        gain_auto=CameraFeatureCapability(True, True, True),
        frame_rate=CameraFeatureCapability(True, True, True),
        width=CameraFeatureCapability(True, True, True),
        height=CameraFeatureCapability(True, True, True),
        offset_x=CameraFeatureCapability(True, True, True),
        offset_y=CameraFeatureCapability(True, True, True),
        pixel_format=CameraFeatureCapability(True, True, True),
        trigger_mode=CameraFeatureCapability(True, True, True),
        acquisition_mode=CameraFeatureCapability(True, True, True, "Continuous"),
    )

