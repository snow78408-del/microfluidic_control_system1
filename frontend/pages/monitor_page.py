from __future__ import annotations

import base64
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from backend.orchestrator.state import SystemState
except Exception:  # pragma: no cover
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
        self._video_photo = None

        self.err_var = tk.StringVar(value="--")
        self.adjust_var = tk.StringVar(value="--")
        self.freeze_var = tk.StringVar(value="--")
        self.stop_var = tk.StringVar(value="--")
        self.q1_cmd_var = tk.StringVar(value="--")
        self.q2_cmd_var = tk.StringVar(value="--")
        self.q1_actual_var = tk.StringVar(value="--")
        self.q2_actual_var = tk.StringVar(value="--")
        self.ch1_exec_var = tk.StringVar(value="--")
        self.ch2_exec_var = tk.StringVar(value="--")
        self.reason_var = tk.StringVar(value="--")

        self.video_mode_var = tk.StringVar(value="--")
        self.video_source_var = tk.StringVar(value="--")
        self.video_res_var = tk.StringVar(value="--")

        self._canvas: tk.Canvas | None = None
        self._content: ttk.Frame | None = None
        self._scrollbar: ttk.Scrollbar | None = None
        self._content_window_id: int | None = None

        self._build()

    def _build(self) -> None:
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._content = ttk.Frame(self._canvas)
        self._content_window_id = self._canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._content.bind("<Configure>", self._on_content_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()

        top = ttk.Frame(self._content)
        top.pack(fill="x", padx=10, pady=(6, 4))
        ttk.Button(top, text="参数页", command=lambda: self.app.show_page("parameter")).pack(side="left")
        ttk.Button(top, text="状态页", command=lambda: self.app.show_page("status")).pack(side="left", padx=4)

        self.buttons = ControlButtons(self._content)
        self.buttons.pack(fill="x", padx=10, pady=(2, 6))
        self.buttons.bind_actions(
            on_init=self._on_init,
            on_start=self._on_start,
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_stop=self._on_stop,
        )

        video_frame = ttk.LabelFrame(self._content, text="视频画面")
        video_frame.pack(fill="x", padx=10, pady=6)

        self.video_label = ttk.Label(video_frame, text="等待视频帧...")
        self.video_label.pack(anchor="w", padx=6, pady=6)

        meta = ttk.Frame(video_frame)
        meta.pack(fill="x", padx=6, pady=2)
        ttk.Label(meta, text="视频模式:").grid(row=0, column=0, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_mode_var).grid(row=0, column=1, sticky="w", padx=3, pady=1)
        ttk.Label(meta, text="视频源:").grid(row=0, column=2, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_source_var).grid(row=0, column=3, sticky="w", padx=3, pady=1)
        ttk.Label(meta, text="分辨率:").grid(row=0, column=4, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_res_var).grid(row=0, column=5, sticky="w", padx=3, pady=1)

        info_row = ttk.Frame(self._content)
        info_row.pack(fill="x", padx=10, pady=6)

        self.recognition_panel = RecognitionPanel(info_row)
        self.recognition_panel.pack(side="left", fill="both", expand=True, padx=3)

        self.pump_panel = PumpPanel(info_row)
        self.pump_panel.pack(side="left", fill="both", expand=True, padx=3)

        self.status_panel = StatusPanel(info_row)
        self.status_panel.pack(side="left", fill="both", expand=True, padx=3)

        ctrl_frame = ttk.LabelFrame(self._content, text="PID 控制结果")
        ctrl_frame.pack(fill="x", padx=10, pady=(6, 10))
        rows = [
            ("直径误差", self.err_var),
            ("PID 调节量", self.adjust_var),
            ("反馈冻结", self.freeze_var),
            ("建议停机", self.stop_var),
            ("Q1 指令", self.q1_cmd_var),
            ("Q1 实时灌注速度", self.q1_actual_var),
            ("CH1执行状态", self.ch1_exec_var),
            ("Q2 指令", self.q2_cmd_var),
            ("Q2 实时灌注速度", self.q2_actual_var),
            ("CH2执行状态", self.ch2_exec_var),
            ("原因", self.reason_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(ctrl_frame, text=f"{name}:").grid(row=i, column=0, padx=4, pady=2, sticky="w")
            ttk.Label(ctrl_frame, textvariable=var).grid(row=i, column=1, padx=4, pady=2, sticky="w")
        ctrl_frame.columnconfigure(1, weight=1)

    def _on_content_configure(self, _event=None) -> None:
        if self._canvas is None:
            return
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        if self._canvas is None or self._content_window_id is None:
            return
        self._canvas.itemconfigure(self._content_window_id, width=event.width)

    def _bind_mousewheel(self) -> None:
        if self._canvas is None:
            return

        def _on_wheel(event):
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = int(-event.delta / 120)
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            if delta != 0 and self._canvas is not None:
                self._canvas.yview_scroll(delta, "units")

        self._canvas.bind_all("<MouseWheel>", _on_wheel)
        self._canvas.bind_all("<Button-4>", _on_wheel)
        self._canvas.bind_all("<Button-5>", _on_wheel)

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

    def _set_video_image(self, frame_png_base64: str | None) -> None:
        if not frame_png_base64:
            return
        try:
            self._video_photo = tk.PhotoImage(data=frame_png_base64)
            self.video_label.configure(image=self._video_photo, text="")
        except Exception:
            try:
                raw = base64.b64decode(frame_png_base64)
                b64 = base64.b64encode(raw).decode("ascii")
                self._video_photo = tk.PhotoImage(data=b64)
                self.video_label.configure(image=self._video_photo, text="")
            except Exception:
                self.video_label.configure(text="视频帧解码失败")

    @staticmethod
    def _parse_channel_status(reason: str, channel: int) -> str:
        if not reason:
            return "--"
        text = reason.upper()
        tag = f"CH{channel}"
        if tag not in text:
            return "--"
        fail_keys = ("失败", "FAIL", "ERROR", "异常")
        if any(k in text for k in fail_keys):
            return "失败"
        return "已执行"

    def _poll_once(self) -> None:
        snap = self.app.orchestrator.get_snapshot()
        self.recognition_panel.update_snapshot(snap)
        self.pump_panel.update_snapshot(snap)
        self.status_panel.update_snapshot(snap)
        if snap.pump_state is not None:
            self.q1_actual_var.set(f"{float(snap.pump_state.q1):.6f}")
            self.q2_actual_var.set(f"{float(snap.pump_state.q2):.6f}")
        else:
            self.q1_actual_var.set("--")
            self.q2_actual_var.set("--")

        rec = snap.recognition
        if rec is not None:
            self.video_mode_var.set(rec.video_source_type or "--")
            self.video_source_var.set(rec.video_source or "--")
            if rec.frame_width > 0 and rec.frame_height > 0:
                self.video_res_var.set(f"{rec.frame_width} x {rec.frame_height}")
            else:
                self.video_res_var.set("--")
            self._set_video_image(rec.frame_png_base64)

        ctrl = snap.control
        if ctrl is not None:
            self.err_var.set(f"{ctrl.diameter_error:.6f}")
            self.adjust_var.set(f"{ctrl.adjustment:.6f}")
            self.freeze_var.set("是" if ctrl.freeze_feedback else "否")
            self.stop_var.set("是" if ctrl.suggested_stop else "否")
            self.q1_cmd_var.set(f"{ctrl.q1_command:.6f}")
            self.q2_cmd_var.set(f"{ctrl.q2_command:.6f}")
            self.reason_var.set(ctrl.reason or "--")
            self.ch1_exec_var.set(self._parse_channel_status(ctrl.reason or "", 1))
            self.ch2_exec_var.set(self._parse_channel_status(ctrl.reason or "", 2))

        state_val = snap.system_state.value if hasattr(snap.system_state, "value") else str(snap.system_state)
        self.buttons.update_by_state(SystemState(state_val))
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
            on_error=lambda e: messagebox.showerror("继续失败", str(e)),
        )

    def _on_stop(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.stop,
            on_error=lambda e: messagebox.showerror("停止失败", str(e)),
        )
