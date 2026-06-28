from .alliedvision_camera import AlliedVisionCameraAdapter as _AlliedVisionCameraAdapter


class AlliedVisionCameraAdapter(_AlliedVisionCameraAdapter):
    backend_name = "allied_vision"


__all__ = ["AlliedVisionCameraAdapter"]
