from __future__ import annotations

# Compatibility wrapper. Runtime camera discovery must use
# backend.vision.cameras.registry -> adapters.opencv_adapter.
from .opencv_adapter import OpenCVCameraAdapter

__all__ = ["OpenCVCameraAdapter"]
