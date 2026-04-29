from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import PumpRuntimeState, SystemSnapshot
except Exception:  # pragma: no cover
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
        self.last_update_var = tk.StringVar(value="--")
        self.err_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("串口连接", self.connected_var),
            ("通信状态", self.comm_var),
            ("设备就绪", self.ready_var),
            ("当前 Q1", self.q1_var),
            ("当前 Q2", self.q2_var),
            ("是否灌注中", self.running_var),
            ("最近下发成功", self.last_update_var),
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
        self.ready_var.set("就绪" if state.fully_ready else "未就绪")
        self.q1_var.set(f"{state.q1:.4f}")
        self.q2_var.set(f"{state.q2:.4f}")
        self.running_var.set("运行中" if state.running else "已停止")
        if state.last_update_reason:
            ok_text = "成功" if state.last_update_ok else "失败"
            self.last_update_var.set(f"{ok_text} ({state.last_update_reason})")
        else:
            self.last_update_var.set("--")
        self.err_var.set(state.last_error or "--")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        self.update_pump_state(snapshot.pump_state)
