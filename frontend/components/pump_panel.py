from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ...backend.orchestrator.models import PumpRuntimeState, SystemSnapshot


class PumpPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="泵状态")
        self.connected_var = tk.StringVar(value="--")
        self.comm_var = tk.StringVar(value="--")
        self.ready_var = tk.StringVar(value="--")
        self.q1_var = tk.StringVar(value="--")
        self.q2_var = tk.StringVar(value="--")
        self.running_var = tk.StringVar(value="--")
        self.err_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("串口连接", self.connected_var),
            ("通信建立", self.comm_var),
            ("设备就绪", self.ready_var),
            ("Q1 流速", self.q1_var),
            ("Q2 流速", self.q2_var),
            ("运行状态", self.running_var),
            ("最近错误", self.err_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Label(self, textvariable=var).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        self.columnconfigure(1, weight=1)

    def update_pump_state(self, state: PumpRuntimeState | None) -> None:
        if state is None:
            return
        self.connected_var.set("已连接" if state.connected else "未连接")
        self.comm_var.set("已建立" if state.comm_established else "未建立")
        self.ready_var.set("完全就绪" if state.fully_ready else "未完全就绪")
        self.q1_var.set(f"{state.q1:.4f}")
        self.q2_var.set(f"{state.q2:.4f}")
        self.running_var.set("运行中" if state.running else "已停止")
        self.err_var.set(state.last_error or "--")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        self.update_pump_state(snapshot.pump_state)

