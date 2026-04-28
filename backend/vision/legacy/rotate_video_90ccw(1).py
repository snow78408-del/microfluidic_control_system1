#!/usr/bin/env python3
"""
将输入视频逆时针旋转 90 度并输出到新文件。
用法: python rotate_video_90ccw.py -i <输入视频路径> [-o 输出视频路径]
"""

import argparse
import sys
from typing import Optional

import cv2


def rotate_video_90_ccw(input_path: str, output_path: Optional[str] = None) -> None:
    """
    读取视频，将每一帧逆时针旋转 90 度，写入新视频。

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径，若为 None 则在输入文件名后加 _rotated_90ccw
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}", file=sys.stderr)
        sys.exit(1)

    # 原始尺寸 (宽, 高)，旋转后变为 (高, 宽)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base = input_path.rsplit(".", 1)
        output_path = f"{base[0]}_rotated_90ccw.{base[1]}" if len(base) == 2 else f"{input_path}_rotated_90ccw.mp4"

    # 逆时针 90 度后：宽高互换
    out_w, out_h = h, w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        print(f"错误: 无法创建输出视频 {output_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 逆时针 90 度 = ROTATE_90_COUNTERCLOCKWISE
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            out.write(rotated)
            frame_idx += 1
            if frame_idx % 100 == 0 and total_frames > 0:
                print(f"\r已处理 {frame_idx}/{total_frames} 帧", end="", flush=True)
    finally:
        cap.release()
        out.release()

    print(f"\n完成: 已写入 {output_path} (共 {frame_idx} 帧)")


def main() -> None:
    parser = argparse.ArgumentParser(description="将视频逆时针旋转 90 度")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", default=None, help="输出视频路径（可选）")
    args = parser.parse_args()
    rotate_video_90_ccw(args.input, args.output)


if __name__ == "__main__":
    main()
