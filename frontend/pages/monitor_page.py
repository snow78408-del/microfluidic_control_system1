from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from ...backend.orchestrator.state import SystemState
from ..components.control_buttons import ControlButtons
from ..components.pump_panel import PumpPanel
from ..components.recognition_panel import RecognitionPanel
from ..components.status_panel import StatusPanel


class MonitorPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._poll_job = None
        self.err_var = tk.StringVar(value="--")
        self.adjust_var = tk.StringVar(value="--")
        self.freeze_var = tk.StringVar(value="--")
        self.stop_var = tk.StringVar(value="--")
        self.q1_cmd_var = tk.StringVar(value="--")
        self.q2_cmd_var = tk.StringVar(value="--")
        self.reason_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=16, pady=8)
        ttk.Button(top, text="参数页面", command=lambda: self.app.show_page("parameter")).pack(side="left")
        ttk.Button(top, text="状态页面", command=lambda: self.app.show_page("status")).pack(side="left", padx=6)

        video_frame = ttk.LabelFrame(self, text="视频画面")
        video_frame.pack(fill="x", padx=16, pady=8)
        ttk.Label(video_frame, text="此处为视频展示区域（由视觉模块渲染）").pack(anchor="w", padx=8, pady=8)

        info_row = ttk.Frame(self)
        info_row.pack(fill="x", padx=16, pady=8)

        self.recognition_panel = RecognitionPanel(info_row)
        self.recognition_panel.pack(side="left", fill="both", expand=True, padx=6)

        self.pump_panel = PumpPanel(info_row)
        self.pump_panel.pack(side="left", fill="both", expand=True, padx=6)

        self.status_panel = StatusPanel(info_row)
        self.status_panel.pack(side="left", fill="both", expand=True, padx=6)

        ctrl_frame = ttk.LabelFrame(self, text="控制结果")
        ctrl_frame.pack(fill="x", padx=16, pady=8)
        rows = [
            ("当前直径误差", self.err_var),
            ("当前 PID 调节量", self.adjust_var),
            ("PID 是否冻结", self.freeze_var),
            ("是否建议停机", self.stop_var),
            ("Q1 指令流速", self.q1_cmd_var),
            ("Q2 指令流速", self.q2_cmd_var),
            ("控制说明", self.reason_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(ctrl_frame, text=f"{name}:").grid(row=i, column=0, padx=6, pady=3, sticky="w")
            ttk.Label(ctrl_frame, textvariable=var).grid(row=i, column=1, padx=6, pady=3, sticky="w")
        ctrl_frame.columnconfigure(1, weight=1)

        self.buttons = ControlButtons(self)
        self.buttons.pack(fill="x", padx=16, pady=8)
        self.buttons.bind_actions(
            on_init=self._on_init,
            on_start=self._on_start,
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_stop=self._on_stop,
        )

    def on_show(self) -> None:
        self._start_poll()

    def on_hide(self) -> None:
        self._stop_poll()

    def _start_poll(self) -> None:
        self._stop_poll()
        self._poll_once()

    def _stop_poll(self) -> None:
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _poll_once(self) -> None:
        snap = self.app.orchestrator.get_snapshot()
        self.recognition_panel.update_snapshot(snap)
        self.pump_panel.update_snapshot(snap)
        self.status_panel.update_snapshot(snap)

        ctrl = snap.control
        if ctrl is not None:
            self.err_var.set(f"{ctrl.diameter_error:.6f}")
            self.adjust_var.set(f"{ctrl.adjustment:.6f}")
            self.freeze_var.set("是" if ctrl.freeze_feedback else "否")
            self.stop_var.set("是" if ctrl.suggested_stop else "否")
            self.q1_cmd_var.set(f"{ctrl.q1_command:.6f}")
            self.q2_cmd_var.set(f"{ctrl.q2_command:.6f}")
            self.reason_var.set(ctrl.reason or "--")

        self.buttons.update_by_state(SystemState(snap.system_state))
        self._poll_job = self.after(self.app.refresh_interval_ms, self._poll_once)

    def _on_init(self) -> None:
        self.app.show_page("init")

    def _on_start(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.start,
            on_error=lambda e: messagebox.showerror("启动失败", str(e)),
        )

    def _on_pause(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.pause,
            on_error=lambda e: messagebox.showerror("暂停失败", str(e)),
        )

    def _on_resume(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.resume,
            on_error=lambda e: messagebox.showerror("恢复失败", str(e)),
        )

    def _on_stop(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.stop,
            on_error=lambda e: messagebox.showerror("停止失败", str(e)),
        )

