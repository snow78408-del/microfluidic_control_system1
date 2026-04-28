from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import cv2

RotationMode = Literal["ccw90", "cw90", "180", "auto"]


def rotate_frame(frame, mode: RotationMode):
    if mode == "ccw90":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "cw90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_video(input_path: str, output_path: Optional[str] = None, mode: RotationMode = "ccw90") -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_path is None:
        src = Path(input_path)
        output_path = str(src.with_name(f"{src.stem}_rotated_{mode}.mp4"))

    if mode in ("ccw90", "cw90", "auto"):
        output_size = (height, width)
    else:
        output_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, output_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {output_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rotated = rotate_frame(frame, mode)
            writer.write(rotated)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError("No frame was processed during rotation.")

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video preprocessing: rotate video.")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--mode", choices=["ccw90", "cw90", "180", "auto"], default="ccw90")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = rotate_video(args.input, output_path=args.output, mode=args.mode)
    print(f"Preprocess completed: {output_path}")


if __name__ == "__main__":
    main()
