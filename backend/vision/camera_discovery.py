from __future__ import annotations

from typing import Any, Callable

from .config import CameraDiscoveryConfig, CameraSystemConfig
from .cameras.manager import CameraManager
from .cameras.models import CameraDeviceInfo
from .models import CameraDevice


def _coerce_camera_system_config(config: Any | None) -> CameraSystemConfig:
    if isinstance(config, CameraSystemConfig):
        return config
    result = CameraSystemConfig()
    if config is None:
        return result
    for name in ("mvs_sdk_path", "opencv_scan_indices", "gentl_producer_paths"):
        if hasattr(config, name):
            setattr(result, name, getattr(config, name))
    return result


def _manager(config: CameraDiscoveryConfig | CameraSystemConfig | None, logger: Callable[[str], None] | None) -> CameraManager:
    return CameraManager(config=_coerce_camera_system_config(config), logger=logger)


def _to_legacy_device(device: CameraDeviceInfo) -> CameraDevice:
    return CameraDevice(
        device_id=device.unique_id or device.device_id,
        device_name=device.user_defined_name or device.model or device.device_id,
        device_type=device.device_type,
        transport_type=device.transport_type,
        manufacturer=device.manufacturer,
        model=device.model,
        serial_number=device.serial_number,
        user_defined_name=device.user_defined_name,
        current_ip=device.ip_address,
        available=device.available,
        error=device.error,
        device_path=device.usb_path or device.device_id,
        raw_info=device.to_dict(),
    )


def discover_cameras(
    config: CameraDiscoveryConfig | CameraSystemConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> list[CameraDevice]:
    manager = _manager(config, logger)
    return [_to_legacy_device(device) for device in manager.discover_all()]


def discover_hikrobot_cameras(
    config: CameraDiscoveryConfig | CameraSystemConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> list[CameraDevice]:
    manager = _manager(config, logger)
    return [_to_legacy_device(device) for device in manager.discover_all() if device.backend_name == "hikrobot"]


def discover_usb_cameras(
    config: CameraDiscoveryConfig | CameraSystemConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> list[CameraDevice]:
    manager = _manager(config, logger)
    return [_to_legacy_device(device) for device in manager.discover_all() if device.backend_name == "opencv"]


def discover_standard_cameras(
    config: CameraSystemConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    manager = _manager(config, logger)
    return [device.to_dict() for device in manager.discover_all()]


def discover_cameras_result(
    config: CameraSystemConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manager = _manager(config, logger)
    return manager.discover_all_result().to_dict()
