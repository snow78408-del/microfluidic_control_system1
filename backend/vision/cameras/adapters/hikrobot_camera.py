from __future__ import annotations

import os
import time
from typing import Any

from ...camera_adapters.hikrobot_camera import HikrobotCameraAdapter as LegacyHikrobotAdapter
from ...camera_adapters.hikrobot_camera import HikrobotSdkLoader
from ...models import CameraConfig
from ..base import BaseCameraAdapter
from ..models import (
    CameraCapabilities,
    CameraDeviceInfo,
    CameraFeatureCapability,
    DEVICE_TYPE_INDUSTRIAL,
    FrameData,
)


class HikrobotCameraAdapter(BaseCameraAdapter):
    backend_name = "hikrobot"
    supported_manufacturers = ("HIKROBOT", "Hikrobot", "Hikvision", "Hikrobot Machine Vision")
    backend_priority = 10

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        sdk_path = _sdk_search_path(config)
        self._legacy = LegacyHikrobotAdapter(CameraConfig(mvs_sdk_path=sdk_path), logger=logger, sdk_path=sdk_path)
        self._device: CameraDeviceInfo | None = None

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        sdk_path = _sdk_search_path(config)
        loader = HikrobotSdkLoader(configured_path=sdk_path, extra_paths=getattr(config, "sdk_paths", ()) or ())
        try:
            loader.load()
            return True, "HIKROBOT MVS SDK available"
        except Exception:
            return False, loader.error or "HIKROBOT MVS SDK unavailable"

    @classmethod
    def check_backend(cls, config: Any | None = None, logger=None):
        from ..models import CameraBackendStatus

        sdk_path = _sdk_search_path(config)
        loader = HikrobotSdkLoader(configured_path=sdk_path, extra_paths=getattr(config, "sdk_paths", ()) or (), logger=logger)
        found = loader.find_sdk()
        loaded = False
        if found:
            try:
                loader.load()
                loaded = True
            except Exception:
                loaded = False
        elif logger:
            logger(f"[HIKROBOT][INSTALL] MVS installation found={loader.mvs_installation_found}")
            logger(f"[HIKROBOT][PYTHON] MvCameraControl_class.py path={loader.python_interface_path or ''}")
            logger(f"[HIKROBOT][DLL] MvCameraControl.dll path={loader.dll_path or ''}")
            logger("[HIKROBOT][IMPORT] success=False")
        return CameraBackendStatus(
            backend_name=cls.backend_name,
            display_name="HIKROBOT MVS",
            sdk_loaded=loaded,
            sdk_path=str(loader.sdk_root or ""),
            python_module_loaded=loaded,
            python_module_path=str(loader.import_dir or loader.python_interface_path or ""),
            dll_loaded=bool(loaded and loader.dll_path),
            dll_path=str(loader.dll_path.parent if loader.dll_path else ""),
            cti_loaded=bool(loader.cti_paths),
            cti_paths=[str(path) for path in loader.cti_paths],
            backend_available=loaded,
            error="" if loaded else (loader.error or "HIKROBOT MVS SDK unavailable"),
        )

    @classmethod
    def get_backend_diagnostics(cls, config: Any | None = None) -> dict[str, Any]:
        sdk_path = _sdk_search_path(config)
        loader = HikrobotSdkLoader(configured_path=sdk_path, extra_paths=getattr(config, "sdk_paths", ()) or ())
        found = loader.find_sdk()
        return {
            "display_name": "HIKROBOT MVS",
            "sdk_loaded": found,
            "sdk_path": str(loader.sdk_root or ""),
            "python_module_loaded": False,
            "python_module_path": str(loader.import_dir or loader.python_interface_path or ""),
            "dll_loaded": False,
            "dll_path": str(loader.dll_path.parent if loader.dll_path else ""),
            "cti_loaded": bool(loader.cti_paths),
            "cti_paths": [str(path) for path in loader.cti_paths],
            "error": "" if found else loader.error,
        }
    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        sdk_path = _sdk_search_path(config)
        devices = LegacyHikrobotAdapter.discover_devices(sdk_path=sdk_path, logger=logger)
        result: list[CameraDeviceInfo] = []
        for device in devices:
            unique = _unique_id(device.manufacturer, device.model, device.serial_number, device.current_ip, device.device_id)
            caps = CameraCapabilities(
                exposure=CameraFeatureCapability(True, True, True),
                exposure_auto=CameraFeatureCapability(True, True, True, "Off", available_values=["Off", "Once", "Continuous"]),
                gain=CameraFeatureCapability(True, True, True),
                gain_auto=CameraFeatureCapability(True, True, True, "Off", available_values=["Off", "Once", "Continuous"]),
                frame_rate=CameraFeatureCapability(True, True, True),
                width=CameraFeatureCapability(True, True, True),
                height=CameraFeatureCapability(True, True, True),
                offset_x=CameraFeatureCapability(True, True, True),
                offset_y=CameraFeatureCapability(True, True, True),
                pixel_format=CameraFeatureCapability(True, True, True),
                trigger_mode=CameraFeatureCapability(True, True, True, "Off", available_values=["Off", "On"]),
                packet_size=CameraFeatureCapability(device.transport_type == "GigE", True, True),
                acquisition_mode=CameraFeatureCapability(True, True, True, "Continuous", available_values=["Continuous"]),
            )
            result.append(
                CameraDeviceInfo(
                    device_id=device.device_id,
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=device.manufacturer or "HIKROBOT",
                    model=device.model,
                    serial_number=device.serial_number,
                    user_defined_name=device.user_defined_name,
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=device.transport_type or "Unknown",
                    ip_address=device.current_ip,
                    available=bool(device.available),
                    capabilities=caps,
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    error=device.error,
                    raw_info=device.raw_info,
                )
            )
        return result

    def open(self, device_info: CameraDeviceInfo) -> None:
        self._device = device_info
        self._legacy.open(device_info.device_id)

    def close(self) -> None:
        self._legacy.close()

    def start_stream(self) -> None:
        self._legacy.start_stream()

    def stop_stream(self) -> None:
        self._legacy.stop_stream()

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        deadline = time.time() + max(0.001, timeout_ms / 1000.0)
        packet = None
        while time.time() < deadline:
            packet = self._legacy.read_frame()
            if packet.valid and packet.frame is not None:
                frame = packet.frame
                return FrameData(
                    image=frame,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    pixel_format=self._legacy.get_status().pixel_format,
                    source_backend=self.backend_name,
                    device_unique_id=self._device.unique_id if self._device else "",
                    valid=True,
                )
            time.sleep(0.01)
        error = packet.error if packet is not None else "婵炴潙鍢查幃宥夋儎閸涘﹥绨氶柛娆愮墪閹舵氨鎼鹃崨顔筋槯"
        self._last_error = error
        return FrameData(None, 0, time.time(), source_backend=self.backend_name, valid=False, error=error)

    def get_latest_frame(self) -> FrameData:
        return self.read_frame(timeout_ms=1)

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return self._device.capabilities if self._device else CameraCapabilities()

    def get_feature(self, name: str) -> Any:
        info = self._legacy.get_parameter_info(_feature_name(name))
        return info.current_value

    def set_feature(self, name: str, value: Any) -> None:
        if name in {"exposure", "exposure_time"}:
            self._legacy.set_exposure(float(value))
        elif name == "gain":
            self._legacy.set_gain(float(value))
        elif name == "frame_rate":
            self._legacy.set_frame_rate(float(value))
        elif name in {"width", "height", "offset_x", "offset_y"}:
            current = {
                "width": None,
                "height": None,
                "offset_x": None,
                "offset_y": None,
                name: int(value),
            }
            self._legacy.set_resolution(**current)
        elif name == "trigger_mode":
            self._legacy.set_trigger_mode(str(value))

    def is_open(self) -> bool:
        return self._legacy.is_open()

    def is_streaming(self) -> bool:
        return self._legacy.is_streaming()

    def get_last_error(self) -> str:
        return self._last_error or self._legacy.get_status().last_camera_error


def _unique_id(manufacturer: str, model: str, serial: str, ip: str, fallback: str) -> str:
    if manufacturer and model and serial:
        return f"{manufacturer}:{model}:{serial}"
    if serial:
        return f"HIKROBOT:{serial}"
    if ip:
        return f"HIKROBOT:ip:{ip}"
    return fallback


def _sdk_search_path(config: Any | None) -> str:
    values: list[str] = []
    for name in ("hikrobot_mvs_sdk_path", "mvs_sdk_path"):
        value = str(getattr(config, name, "") or "").strip()
        if value:
            values.append(value)
    values.extend(str(path) for path in (getattr(config, "sdk_paths", ()) or ()) if str(path).strip())
    return os.pathsep.join(dict.fromkeys(values))


def _feature_name(name: str) -> str:
    return {
        "exposure": "ExposureTime",
        "exposure_time": "ExposureTime",
        "gain": "Gain",
        "frame_rate": "AcquisitionFrameRate",
        "width": "Width",
        "height": "Height",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "pixel_format": "PixelFormat",
        "trigger_mode": "TriggerMode",
    }.get(name, name)
