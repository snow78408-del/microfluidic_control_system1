from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ...backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot


class RecognitionPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="识别结果")
        self.count_var = tk.StringVar(value="--")
        self.diameter_var = tk.StringVar(value="--")
        self.single_cell_rate_var = tk.StringVar(value="--")
        self.valid_var = tk.StringVar(value="--")
        self.timestamp_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("液滴总数", self.count_var),
            ("平均直径", self.diameter_var),
            ("单胞率", self.single_cell_rate_var),
            ("当前帧有效", self.valid_var),
            ("时间戳", self.timestamp_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Label(self, textvariable=var).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        self.columnconfigure(1, weight=1)

    def update_recognition(self, rec: RecognitionSnapshot | None) -> None:
        if rec is None:
            return
        self.count_var.set(str(rec.droplet_count))
        self.diameter_var.set(f"{rec.avg_diameter:.4f}")
        self.single_cell_rate_var.set(f"{rec.single_cell_rate:.4f}")
        self.valid_var.set("有效" if rec.valid_for_control else "无效")
        self.timestamp_var.set(f"{rec.timestamp:.3f}")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        self.update_recognition(snapshot.recognition)

