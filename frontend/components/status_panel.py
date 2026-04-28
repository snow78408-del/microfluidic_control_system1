from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ...backend.orchestrator.models import SystemSnapshot


class StatusPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="系统状态")
        self.system_state_var = tk.StringVar(value="--")
        self.stage_var = tk.StringVar(value="--")
        self.message_var = tk.StringVar(value="--")
        self.error_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("系统状态", self.system_state_var),
            ("当前阶段", self.stage_var),
            ("提示信息", self.message_var),
            ("错误信息", self.error_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Label(self, textvariable=var).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        self.columnconfigure(1, weight=1)

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        self.system_state_var.set(getattr(snapshot.system_state, "value", str(snapshot.system_state)))
        self.stage_var.set(getattr(snapshot.system_state, "value", str(snapshot.system_state)))
        self.message_var.set(snapshot.message or "--")
        self.error_var.set(snapshot.error or "--")

