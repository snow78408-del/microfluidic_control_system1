from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot
except Exception:  # pragma: no cover
    from ...backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot


class RecognitionPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="识别结果")
        self.frame_count_var = tk.StringVar(value="--")
        self.new_crossing_var = tk.StringVar(value="--")
        self.total_count_var = tk.StringVar(value="--")
        self.has_droplet_var = tk.StringVar(value="--")
        self.diameter_var = tk.StringVar(value="--")
        self.single_cell_rate_var = tk.StringVar(value="--")
        self.valid_var = tk.StringVar(value="--")
        self.reason_var = tk.StringVar(value="--")
        self.timestamp_var = tk.StringVar(value="--")
        self.video_mode_var = tk.StringVar(value="--")
        self.video_source_var = tk.StringVar(value="--")
        self.resolution_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("当前帧液滴数", self.frame_count_var),
            ("新增通过液滴", self.new_crossing_var),
            ("累计液滴总数", self.total_count_var),
            ("当前帧有液滴", self.has_droplet_var),
            ("当前平均直径", self.diameter_var),
            ("单胞率", self.single_cell_rate_var),
            ("valid_for_control", self.valid_var),
            ("识别原因", self.reason_var),
            ("视频模式", self.video_mode_var),
            ("视频源", self.video_source_var),
            ("视频分辨率", self.resolution_var),
            ("时间戳", self.timestamp_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Label(self, textvariable=var).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        self.columnconfigure(1, weight=1)

    def update_recognition(self, rec: RecognitionSnapshot | None) -> None:
        if rec is None:
            return
        self.frame_count_var.set(str(rec.frame_droplet_count))
        self.new_crossing_var.set(str(rec.new_crossing_count))
        self.total_count_var.set(str(rec.total_droplet_count))
        self.has_droplet_var.set("是" if rec.has_droplet else "否")
        self.diameter_var.set(f"{rec.avg_diameter:.4f}" if rec.avg_diameter is not None else "None")
        self.single_cell_rate_var.set(f"{rec.single_cell_rate:.4f}")
        self.valid_var.set("是" if rec.valid_for_control else "否")
        self.reason_var.set(rec.reason or rec.control_reason or "--")
        self.video_mode_var.set(rec.video_source_type or "--")
        self.video_source_var.set(rec.video_source or "--")
        if rec.frame_width > 0 and rec.frame_height > 0:
            self.resolution_var.set(f"{rec.frame_width} x {rec.frame_height}")
        else:
            self.resolution_var.set("--")
        self.timestamp_var.set(f"{rec.timestamp:.3f}")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        self.update_recognition(snapshot.recognition)
