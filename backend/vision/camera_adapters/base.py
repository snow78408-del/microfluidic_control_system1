from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import CameraConfig, CameraDevice, CameraStatus, CameraTestResult, FramePacket


class CameraAdapterError(RuntimeError):
    def __init__(self, message: str, error_code: int | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class BaseCameraAdapter(ABC):
    @classmethod
    @abstractmethod
    def discover_devices(cls, *args: Any, **kwargs: Any) -> list[CameraDevice]:
        raise NotImplementedError

    @abstractmethod
    def open(self, device_id: str) -> None:
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
    def read_frame(self) -> FramePacket:
        raise NotImplementedError

    @abstractmethod
    def get_device_info(self) -> CameraDevice | None:
        raise NotImplementedError

    @abstractmethod
    def set_exposure(self, exposure_time: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_frame_rate(self, frame_rate: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_resolution(
        self,
        width: int | None = None,
        height: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_trigger_mode(self, trigger_mode: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_streaming(self) -> bool:
        raise NotImplementedError

    def configure(self, camera_config: CameraConfig) -> None:
        if camera_config.exposure_time is not None:
            self.set_exposure(float(camera_config.exposure_time))
        if camera_config.gain is not None:
            self.set_gain(float(camera_config.gain))
        if camera_config.frame_rate is not None:
            self.set_frame_rate(float(camera_config.frame_rate))
        if any(
            value is not None
            for value in (
                camera_config.width,
                camera_config.height,
                camera_config.offset_x,
                camera_config.offset_y,
            )
        ):
            self.set_resolution(
                camera_config.width,
                camera_config.height,
                camera_config.offset_x,
                camera_config.offset_y,
            )
        self.set_trigger_mode(camera_config.trigger_mode or "Off")

    def test_camera(self, device_id: str, frame_count: int = 3) -> CameraTestResult:
        try:
            self.open(device_id)
            self.set_trigger_mode("Off")
            self.start_stream()
            packets: list[FramePacket] = []
            for _ in range(max(1, int(frame_count))):
                packet = self.read_frame()
                if packet.valid and packet.frame is not None:
                    packets.append(packet)
            if len(packets) < frame_count:
                return CameraTestResult(
                    ok=False,
                    device_id=device_id,
                    error=f"测试取帧失败：需要 {frame_count} 帧，实际 {len(packets)} 帧",
                    frames_read=len(packets),
                )
            frame = packets[-1].frame
            status = self.get_status()
            return CameraTestResult(
                ok=True,
                device_id=device_id,
                message="测试取帧成功",
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                pixel_format=status.pixel_format,
                frame_rate=status.frame_rate,
                frames_read=len(packets),
            )
        except CameraAdapterError as exc:
            return CameraTestResult(ok=False, device_id=device_id, error=str(exc), error_code=exc.error_code)
        except Exception as exc:
            return CameraTestResult(ok=False, device_id=device_id, error=str(exc))
        finally:
            self.stop_stream()
            self.close()

    @abstractmethod
    def get_status(self) -> CameraStatus:
        raise NotImplementedError

