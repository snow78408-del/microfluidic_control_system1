from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import CameraBackendStatus, CameraCapabilities, CameraDeviceInfo, FrameData


class CameraBackendError(RuntimeError):
    def __init__(self, message: str, error_code: int | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class BaseCameraAdapter(ABC):
    backend_name: str = "base"
    supported_manufacturers: tuple[str, ...] = ()
    backend_priority: int = 100

    def __init__(self, config: Any | None = None, logger=None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self._last_error = ""

    @classmethod
    @abstractmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        raise NotImplementedError

    @classmethod
    def check_backend(cls, config: Any | None = None) -> CameraBackendStatus:
        ok, reason = cls.is_backend_available(config)
        status = CameraBackendStatus(
            backend_name=cls.backend_name,
            display_name=cls.backend_name,
            backend_available=bool(ok),
            sdk_loaded=bool(ok),
            python_module_loaded=bool(ok),
            error="" if ok else str(reason or ""),
        )
        diagnostic = getattr(cls, "get_backend_diagnostics", None)
        if callable(diagnostic):
            data = diagnostic(config)
            for key, value in data.items():
                if hasattr(status, key):
                    setattr(status, key, value)
        return status

    @abstractmethod
    def open(self, device_info: CameraDeviceInfo) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_stream(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_stream(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        raise NotImplementedError

    @abstractmethod
    def get_latest_frame(self) -> FrameData:
        raise NotImplementedError

    @abstractmethod
    def get_device_info(self) -> CameraDeviceInfo | None:
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        raise NotImplementedError

    @abstractmethod
    def get_feature(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set_feature(self, name: str, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_streaming(self) -> bool:
        raise NotImplementedError

    def get_last_error(self) -> str:
        return self._last_error
