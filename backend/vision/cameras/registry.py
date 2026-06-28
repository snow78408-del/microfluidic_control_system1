from __future__ import annotations

import logging
import traceback
from typing import Any

from .base import BaseCameraAdapter
from .models import CameraBackendStatus, CameraDeviceInfo, CameraDiscoveryResult, DEVICE_TYPE_USB


class CameraAdapterRegistry:
    def __init__(self, config: Any | None = None, logger=None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self._adapters: dict[str, type[BaseCameraAdapter]] = {}
        self._import_errors: dict[str, str] = {}

    def register_adapter(self, adapter_cls: type[BaseCameraAdapter]) -> None:
        self._adapters[adapter_cls.backend_name] = adapter_cls
        self._log(f"[CAMERA][REGISTRY] registered={adapter_cls.backend_name}")

    def unregister_adapter(self, backend_name: str) -> None:
        self._adapters.pop(backend_name, None)

    def get_available_adapters(self) -> dict[str, tuple[type[BaseCameraAdapter], bool, str]]:
        result: dict[str, tuple[type[BaseCameraAdapter], bool, str]] = {}
        enabled = set(getattr(self.config, "enabled_camera_backends", []) or [])
        for name, adapter_cls in self._adapters.items():
            if enabled and name not in enabled:
                result[name] = (adapter_cls, False, "后端未启用")
                continue
            ok, reason = adapter_cls.is_backend_available(self.config)
            result[name] = (adapter_cls, bool(ok), str(reason or ""))
            tag = "AVAILABLE" if ok else "UNAVAILABLE"
            self._log(f"[CAMERA][BACKEND][{tag}] backend={name} reason={reason}")
        return result

    def get_adapter_class(self, backend_name: str) -> type[BaseCameraAdapter]:
        if backend_name not in self._adapters:
            raise KeyError(f"未知相机后端: {backend_name}")
        return self._adapters[backend_name]

    def get_adapter_for_device(
        self,
        device: CameraDeviceInfo,
        backend_name: str | None = None,
    ) -> type[BaseCameraAdapter]:
        selected = backend_name or device.selected_backend or device.backend_name
        return self.get_adapter_class(selected)

    def discover_from_all_adapters(self) -> list[CameraDeviceInfo]:
        return self.discover_from_all_adapters_with_status().raw_devices

    def discover_from_all_adapters_with_status(self) -> CameraDiscoveryResult:
        self._log("[CAMERA][DISCOVERY][START] all_backends")
        devices: list[CameraDeviceInfo] = []
        statuses: list[CameraBackendStatus] = []
        errors: list[str] = []

        for name, import_error in self._import_errors.items():
            statuses.append(
                CameraBackendStatus(
                    backend_name=name,
                    display_name=name,
                    backend_available=False,
                    error=import_error.splitlines()[-1] if import_error else "adapter import failed",
                    traceback=import_error,
                )
            )

        enabled = set(getattr(self.config, "enabled_camera_backends", []) or [])
        for name, adapter_cls in self._adapters.items():
            self._log(f"[CAMERA][DISCOVERY] backend={name} start")
            if enabled and name not in enabled:
                status = CameraBackendStatus(
                    backend_name=name,
                    display_name=name,
                    backend_available=False,
                    error="backend disabled by config",
                )
                self._log(f"[CAMERA][BACKEND] name={name} available=False reason={status.error}")
                statuses.append(status)
                continue
            try:
                try:
                    status = adapter_cls.check_backend(self.config, logger=self._log)
                except TypeError:
                    status = adapter_cls.check_backend(self.config)
            except Exception as exc:
                logging.exception("camera backend check failed: %s", name)
                status = CameraBackendStatus(
                    backend_name=name,
                    display_name=name,
                    backend_available=False,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            reason = status.error or ""
            self._log(f"[CAMERA][BACKEND] name={name} available={status.backend_available} reason={reason}")
            if not status.backend_available:
                statuses.append(status)
                continue
            try:
                raw = adapter_cls.discover_devices(self.config, logger=self._log)
                status.raw_device_count = len(raw)
                status.final_device_count = len(raw)
                if name == "hikrobot":
                    status.enum_return_code = "0x00000000"
                    for item in raw:
                        if "enum_failed" in item.device_id:
                            status.enum_return_code = item.error
                else:
                    status.enum_return_code = status.enum_return_code or "OK"
                for device in raw:
                    if not device.selected_backend:
                        device.selected_backend = name
                    if not device.available_backends:
                        device.available_backends = [name]
                    device.backend_priority = adapter_cls.backend_priority
                    devices.append(device)
                    self._log(
                        "[CAMERA][DEVICE][FOUND] "
                        f"backend={name} vendor={device.manufacturer} model={device.model} "
                        f"sn={device.serial_number} transport={device.transport_type}"
                    )
            except Exception as exc:
                logging.exception("camera backend discovery failed: %s", name)
                tb = traceback.format_exc()
                status.error = str(exc)
                status.traceback = tb
                errors.append(f"{name}: {exc}")
                self._log(f"[CAMERA][BACKEND][UNAVAILABLE] backend={name} error={exc}")
                continue
            finally:
                self._log(f"[CAMERA][DISCOVERY] backend={name} device_count={status.raw_device_count}")
                statuses.append(status)
        return CameraDiscoveryResult(
            devices=list(devices),
            raw_devices=list(devices),
            deduplicated_devices=[],
            backend_statuses=statuses,
            errors=errors,
            raw_device_count=len(devices),
            final_device_count=len(devices),
        )


def default_registry(config: Any | None = None, logger=None) -> CameraAdapterRegistry:
    registry = CameraAdapterRegistry(config=config, logger=logger)
    adapter_specs = (
        ("hikrobot", ".adapters.hikrobot_adapter", "HikrobotCameraAdapter"),
        ("basler", ".adapters.basler_adapter", "BaslerCameraAdapter"),
        ("daheng", ".adapters.daheng_adapter", "DahengCameraAdapter"),
        ("flir", ".adapters.flir_adapter", "FlirCameraAdapter"),
        ("allied_vision", ".adapters.allied_vision_adapter", "AlliedVisionCameraAdapter"),
        ("gentl", ".adapters.gentl_adapter", "GenTLCameraAdapter"),
        ("opencv", ".adapters.opencv_adapter", "OpenCVCameraAdapter"),
    )
    import importlib

    for backend_name, module_name, class_name in adapter_specs:
        try:
            module = importlib.import_module(module_name, package=__package__)
            registry.register_adapter(getattr(module, class_name))
        except Exception as exc:
            registry._import_errors[backend_name] = traceback.format_exc()
            logging.exception("camera adapter import failed: %s", backend_name)
            if logger:
                logger(f"[CAMERA][BACKEND][UNAVAILABLE] backend={backend_name} error={exc}")
    return registry


def device_dedupe_key(device: CameraDeviceInfo) -> str:
    manufacturer = (device.manufacturer or "").strip().lower()
    model = (device.model or "").strip().lower()
    serial = (device.serial_number or "").strip().lower()
    if manufacturer and model and serial:
        return f"mms:{manufacturer}:{model}:{serial}"
    if serial:
        return f"sn:{manufacturer}:{serial}"
    if device.usb_path:
        return f"usb:{device.usb_path}"
    if device.ip_address:
        return f"ip:{device.ip_address}"
    return f"{device.device_type}:{device.unique_id or device.device_id}"


def backend_sort_key(device: CameraDeviceInfo, preferred_order: list[str]) -> tuple[int, int]:
    try:
        idx = preferred_order.index(device.backend_name)
    except ValueError:
        idx = 99
    opencv_penalty = 10 if device.device_type == DEVICE_TYPE_USB else 0
    return (idx + opencv_penalty, device.backend_priority)
