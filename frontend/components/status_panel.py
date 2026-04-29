from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import SystemSnapshot
except Exception:  # pragma: no cover
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
            ("状态", self.system_state_var),
            ("阶段", self.stage_var),
            ("提示", self.message_var),
            ("错误", self.error_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(self, textvariable=var, wraplength=260).grid(row=i, column=1, sticky="w", padx=4, pady=2)
        self.columnconfigure(1, weight=1)

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        value = getattr(snapshot.system_state, "value", str(snapshot.system_state))
        self.system_state_var.set(value)
        self.stage_var.set(value)
        self.message_var.set(snapshot.message or "--")
        self.error_var.set(snapshot.error or "--")
