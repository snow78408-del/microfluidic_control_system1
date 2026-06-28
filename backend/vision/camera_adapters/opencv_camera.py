from __future__ import annotations

# Compatibility wrapper kept for legacy imports. Runtime camera discovery must use
# backend.vision.cameras.registry -> adapters.opencv_adapter.
from ..cameras.adapters.opencv_adapter import OpenCVCameraAdapter

__all__ = ["OpenCVCameraAdapter"]
